import json
import time
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from pyodsp.alg.bm.cuts import OptimalityCut, FeasibilityCut

from ..node._logger import ILogger
from ..node._node import INode, INodeRoot, INodeLeaf, INodeInner
from ..node._message import (
    DnMessage,
    UpMessage,
    NodeIdx,
)
from ..utils import create_directory


from pyodsp.alg import params as _params
from pyodsp.alg.params import SDDP_SEED


SIMULATION_FILE = "simulation.csv"
SIMULATION_SAMPLES_FILE = "simulation_samples.csv"
STOPPING_FILE = "stopping.json"

# Why a run stopped. "gap" and "stable" are decided at a convergence test,
# "stall" at any iteration, "max_iteration" when the loop runs out.
STOP_GAP = "gap"
STOP_STABLE = "stable"
STOP_STALL = "stall"
STOP_MAX_ITERATION = "max_iteration"

# The SDDP progress table: column names once, then numbers. "sample" and the
# confidence interval are blank except on the iterations that run a
# convergence test (every sample_frequency). The interval is last because it
# is the only variable-width column.
_SDDP_HEADER = f"{'iter':>6}  {'bound':>13}  {'sample':>13}  {'elapsed':>8}  CI"


def _sddp_row(
    iteration: int,
    bound: float,
    elapsed: float,
    sample: float | None = None,
    ci: tuple[float, float] | None = None,
) -> str:
    sample_cell = "-" if sample is None else f"{sample:.4f}"
    ci_cell = "-" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    return (
        f"{iteration:>6}  {bound:>13.4f}  {sample_cell:>13}  "
        f"{elapsed:>8.1f}  {ci_cell}"
    )


@dataclass
class SimulationRound:
    """One convergence test: the bound, against a simulated interval."""

    iteration: int
    bound: float
    mean: float
    lower: float
    upper: float
    sample_size: int
    confidence_level: float


class Lattice:
    def __init__(
        self,
        nodes: List[List[INode]],
        logger: ILogger,
        filedir: Path,
        max_iteration: int = 1000,
        sample_frequency: int = 10,
        sample_size: int = 1000,
        confidence_level: float = 0.95,
        gap_tolerance: float | None = None,
        stable_tolerance: float | None = None,
        stall_iterations: int | None = None,
        stall_tolerance: float | None = None,
    ) -> None:
        """
        Args:
            gap_tolerance: Stop when the simulated cost's upper confidence
                limit is within this fraction of the bound.
            stable_tolerance: Stop when re-simulating the previous test's
                paths changes their mean cost by less than this fraction —
                the whole confidence interval of the change inside
                +/- tolerance — or when no path's cost changed at all.
            stall_iterations, stall_tolerance: Stop when the bound has
                moved by less than stall_tolerance (relative) over the last
                stall_iterations iterations. 0 iterations turns it off.

        Each defaults to its SDDP_* value in pyodsp.alg.params.
        """
        self.num_stages = len(nodes)
        self._verify_nodes(nodes)
        self.logger = logger
        self.filedir = filedir
        self.max_iteration = max_iteration
        self.sample_frequency = sample_frequency
        self.sample_size = sample_size
        self.confidence_level = confidence_level
        self.gap_tolerance = _default(gap_tolerance, _params.SDDP_GAP_TOLERANCE)
        self.stable_tolerance = _default(
            stable_tolerance, _params.SDDP_STABLE_TOLERANCE
        )
        self.stall_iterations = int(
            _default(stall_iterations, _params.SDDP_STALL_ITERATIONS)
        )
        self.stall_tolerance = _default(stall_tolerance, _params.SDDP_STALL_TOLERANCE)
        create_directory(self.filedir)

        self._last_dn_messages: Dict[NodeIdx, DnMessage] = {}

        self.prev_samples = None
        self._start_time = 0.0
        # The latest deterministic bound, in the models' own units. run()
        # reports it at the end -- SDDP leaves a policy, not a proven optimum,
        # so this is a bound and not a "final objective value".
        self.bound: float | None = None
        # One entry per convergence test. SDDP's bound is deterministic but
        # the other side is estimated by simulation, so what it has against
        # the bound is an interval rather than an incumbent — recorded here
        # so a finished run can report and plot it, not just log it.
        self.simulation_rounds: List[SimulationRound] = []
        # The individual draws behind the most recent round. Only the last
        # is kept: it is the converged policy's cost distribution, and
        # every round before it describes a policy that no longer exists.
        self.simulation_samples: List[float] = []
        # The bound at every iteration, internal (minimize) units — what
        # the stall test looks back over.
        self.bound_history: List[float] = []
        self.stop_reason: str | None = None
        self.stop_message: str | None = None

    def _verify_nodes(self, nodes: List[List[INode]]) -> None:
        self.root: INodeRoot | None = None
        self.leaves: List[INodeLeaf] = []
        self.nodes: Dict[NodeIdx, INode] = {}
        self.stages: Dict[int, List[NodeIdx]] = {k: [] for k in range(self.num_stages)}
        for stage in range(self.num_stages):
            for node in nodes[stage]:
                self.nodes[node.get_idx()] = node
                self.stages[stage].append(node.get_idx())
            if stage == 0:
                if len(nodes[stage]) > 1:
                    raise ValueError(
                        f"Number of nodes is {len(nodes[stage])} in stage {stage}."
                    )
                node = nodes[stage][0]
                if isinstance(node, INodeLeaf):
                    raise ValueError(f"Stage {stage} must be root node.")
                assert isinstance(node, INodeRoot)
                assert not isinstance(node, INodeLeaf)
                self.root = node
            elif stage == self.num_stages - 1:
                for node in nodes[stage]:
                    if isinstance(node, INodeRoot):
                        raise ValueError(f"Stage {stage} must be leaf node.")
                    assert isinstance(node, INodeLeaf)
                    assert not isinstance(node, INodeRoot)
                    self.leaves.append(node)
            else:
                for node in nodes[stage]:
                    if not isinstance(node, INodeInner):
                        raise ValueError(f"Stage {stage} must be inner node.")

    def run(self) -> None:
        """Run SDDP to convergence.

        Unlike Tree/HubAndSpoke, this takes no initial solution: each
        iteration's forward pass starts from the root's own solve, so
        there is nowhere for a caller-supplied one to enter.
        """
        self.logger.log_initialization(
            sample_frequency=self.sample_frequency,
            sample_size=self.sample_size,
            confidence=self.confidence_level,
        )
        self._run_init()
        self._run_main()
        self.logger.log_completion(self.bound, label="Bound")
        self._save()

    def _run_init(self) -> None:
        if self.root is None:
            raise ValueError("Root node not found")
        self.root.set_depth(0)
        for stage in range(self.num_stages - 1):
            self._run_init_forward(stage)

        for stage in range(self.num_stages - 1, 0, -1):
            self._run_init_backward(stage)

    def _quiet_step_logging(self, node: INodeRoot) -> None:
        """Drop a stage node's bundle-method chatter to DEBUG.

        SDDP runs each node one bundle step per visit and resets it in
        between, so the step-by-step "completed / Total iterations: 1 /
        Final objective value" lifecycle repeats every visit -- thousands of
        lines for a short run. The lattice narrates progress at INFO instead;
        the per-step detail stays available at DEBUG.
        """
        bm = getattr(node.alg_root, "bm", None)
        bm_logger = getattr(bm, "logger", None)
        if bm_logger is not None:
            bm_logger.per_step = True

    def _run_init_forward(self, stage: int) -> None:
        assert stage < self.num_stages - 1
        for node_idx in self.stages[stage]:
            node = self.nodes[node_idx]
            assert isinstance(node, INodeRoot)
            node.set_logger()
            self._quiet_step_logging(node)

        node = self.nodes[
            self.stages[stage][0]
        ]  # get first node in stage as representative
        assert isinstance(node, INodeRoot)
        init_dn_message = node.get_init_dn_message()

        for child_idx in self.stages[stage + 1]:
            child = self.nodes[child_idx]
            assert isinstance(child, INodeLeaf)
            child.pass_init_dn_message(init_dn_message)

    def _run_init_backward(self, stage: int) -> None:
        assert stage > 0
        init_up_messages = {}
        for child_id in self.stages[stage]:
            child = self.nodes[child_id]
            assert isinstance(child, INodeLeaf)
            child.build()
            init_up_message = child.get_init_up_message()
            init_up_messages[child_id] = init_up_message

        for node_idx in self.stages[stage - 1]:
            node = self.nodes[node_idx]
            assert isinstance(node, INodeRoot)
            node.pass_init_up_messages(init_up_messages)
            node.build()
            node.reset()

    def _run_main(self) -> None:
        if self.root is None:
            raise ValueError("Root node not found")
        multiplier = self.root.get_sense_multiplier()
        self._start_time = time.time()
        self.logger.log_info(_SDDP_HEADER)
        bound = -1e9
        for iteration in range(self.max_iteration):
            bound = self._run_root()
            self.bound = bound * multiplier
            if self._bound_stalled(bound):
                self._final_round(bound, iteration)
                break
            if iteration % self.sample_frequency == self.sample_frequency - 1:
                if self._termination(bound, iteration):
                    break
            else:
                self.logger.log_info(
                    _sddp_row(
                        iteration + 1,
                        self.bound,
                        time.time() - self._start_time,
                    )
                )
                self._run_forwards(self._iteration_rng(iteration))

            self._run_backwards()
        else:
            self._stop(STOP_MAX_ITERATION, "the iteration limit was reached")

    def _sample_rng(self, sample_idx: int) -> np.random.Generator:
        """The generator for Monte Carlo sample `sample_idx`.

        Keyed by the *global* sample index rather than by call order, so a
        sample follows the same scenario path no matter which round it is
        drawn in (_termination's prev_samples comparison is only
        meaningful if it does) and no matter which rank runs it (see
        LatticeMpi, whose results are therefore independent of rank
        count).

        spawn_key addresses the same child SeedSequence that
        SeedSequence(SDDP_SEED).spawn(n)[sample_idx] would produce, but
        directly, without spawning the whole list. Unlike seeding with
        consecutive integers, children of a SeedSequence are
        statistically independent by construction.
        """
        return np.random.default_rng(
            np.random.SeedSequence(SDDP_SEED, spawn_key=(sample_idx,))
        )

    def _iteration_rng(self, iteration: int) -> np.random.Generator:
        """The generator for the trunk forward pass of `iteration`, offset
        past the sample indices so a trunk pass never replays a simulation
        stream.
        """
        return np.random.default_rng(
            np.random.SeedSequence(SDDP_SEED, spawn_key=(self.sample_size + iteration,))
        )

    def _collect_samples(self) -> List[float]:
        """The Monte Carlo sampling loop used to estimate the objective's
        confidence interval in _termination. Each sample only depends on
        the current (frozen) set of cuts — this is the embarrassingly
        parallel step LatticeMpi distributes across ranks.
        """
        objectives = []
        for i in range(self.sample_size):
            objective = self._run_forwards(self._sample_rng(i))
            objectives.append(objective)
        return objectives

    def _confidence_interval(self, samples: List[float]) -> tuple[float, float]:
        """Student-t confidence interval of the sample mean. When every
        sample is identical the standard error is zero and scipy would
        return (nan, nan) — the interval then collapses onto the mean.
        """
        mean = float(np.mean(samples))
        scale = float(st.sem(samples))
        if not scale > 0.0:
            return mean, mean
        ci_d, ci_u = st.t.interval(
            confidence=self.confidence_level,
            df=len(samples) - 1,
            loc=mean,
            scale=scale,
        )
        return float(ci_d), float(ci_u)

    def _simulate(self, bound: float, iteration: int) -> List[float]:
        """Run one convergence test's simulation, record it and log it."""
        objectives = self._collect_samples()
        ci_d, ci_u = self._confidence_interval(objectives)
        self._record_simulation(iteration, objectives, ci_d, ci_u, bound)
        rnd = self.simulation_rounds[-1]
        self.logger.log_info(
            _sddp_row(
                iteration + 1,
                rnd.bound,
                time.time() - self._start_time,
                sample=rnd.mean,
                ci=(rnd.lower, rnd.upper),
            )
        )
        return objectives

    def _termination(self, bound: float, iteration: int = -1) -> bool:
        """A convergence test: simulate, then apply the gap and the
        stability rules.

        Gap. Always a minimization here — PyomoSolver converts a maximize
        model on construction — so the bound is a lower bound and the
        sample mean's upper confidence limit the side that meets it. The
        test is one-sided: an upper limit already below the bound (sampling
        noise) has met it too.

        Stability. The samples are keyed by index (see _sample_rng), so
        each test re-draws the previous test's paths and the change in
        cost is paired. The run stops only when that change is
        indistinguishable from zero in *both* directions. A one-sided test
        on improvement alone also passes when the policy got worse on these
        paths — which it routinely does while cuts are still being added —
        and stops runs whose bound is still climbing.
        """
        objectives = self._simulate(bound, iteration)
        ci_u = self._confidence_interval(objectives)[1]

        scale = max(abs(bound), abs(ci_u), 1e-12)
        if ci_u - bound <= self.gap_tolerance * scale:
            self._stop(
                STOP_GAP,
                f"the simulated cost's upper confidence limit {ci_u:.6g} is "
                f"within {self.gap_tolerance:g} of the bound {bound:.6g}",
            )
            return True

        if self.prev_samples is not None:
            sample_diffs = [
                self.prev_samples[i] - objectives[i] for i in range(len(objectives))
            ]
            if all(abs(diff) <= 1e-9 for diff in sample_diffs):
                self._stop(
                    STOP_STABLE,
                    "no simulated path's cost changed since the last test",
                )
                return True
            diff_lo, diff_hi = self._confidence_interval(sample_diffs)
            band = self.stable_tolerance * max(abs(float(np.mean(objectives))), 1e-12)
            if -band <= diff_lo and diff_hi <= band:
                self._stop(
                    STOP_STABLE,
                    f"the simulated cost changed by [{-diff_hi:.4g}, "
                    f"{-diff_lo:.4g}] since the last test, within "
                    f"{self.stable_tolerance:g} of its mean",
                )
                return True

        self.prev_samples = objectives
        return False

    def _bound_stalled(self, bound: float) -> bool:
        """Record this iteration's bound; True once it has moved by less
        than stall_tolerance (relative) over the last stall_iterations.

        The bound is the one side SDDP computes exactly, and it only rises
        as cuts are added, so a plateau in it is a deterministic signal
        that the cuts have stopped adding information.
        """
        self.bound_history.append(bound)
        window = self.stall_iterations
        if window <= 0 or len(self.bound_history) <= window:
            return False
        before = self.bound_history[-1 - window]
        scale = max(abs(bound), abs(before), 1e-12)
        return abs(bound - before) <= self.stall_tolerance * scale

    def _final_round(self, bound: float, iteration: int) -> None:
        """Stopping on a stalled bound: simulate the final policy once, so
        the run still ends with an interval for it, then record why."""
        # (never a duplicate: the stall test runs before this iteration's
        # convergence test would)
        self._simulate(bound, iteration)
        window = self.stall_iterations
        self._stop(
            STOP_STALL,
            f"the bound moved by less than {self.stall_tolerance:g} over the "
            f"last {window} iterations",
        )

    def _stop(self, reason: str, message: str) -> None:
        self.stop_reason = reason
        self.stop_message = message
        self.logger.log_info(f"SDDP terminated ({reason}): {message}")

    def _record_simulation(
        self,
        iteration: int,
        objectives: List[float],
        ci_d: float,
        ci_u: float,
        bound: float,
    ) -> None:
        """Keep one convergence test, in the units the models were written in.

        The samples are in the internal minimize convention, like every
        other objective value the algorithms handle; the root knows what
        converting them back means.
        """
        assert self.root is not None
        multiplier = self.root.get_sense_multiplier()
        lower, upper = ci_d * multiplier, ci_u * multiplier
        self.simulation_samples = [value * multiplier for value in objectives]
        self.simulation_rounds.append(
            SimulationRound(
                iteration=iteration,
                bound=bound * multiplier,
                mean=float(np.mean(objectives)) * multiplier,
                # Negating swaps which limit is which.
                lower=min(lower, upper),
                upper=max(lower, upper),
                sample_size=len(objectives),
                confidence_level=self.confidence_level,
            )
        )

    def _run_root(self) -> float:
        assert self.root is not None
        self._run_forward(self.root)

        return self.root.alg_root.bm.get_objective_value()  # FIXME: properly access

    def _run_forwards(self, rng: np.random.Generator) -> float:
        node = self.root
        assert node is not None
        path = [node.get_idx()]
        for stage in range(1, self.num_stages):
            # randomly sample node in the next stage
            prob = [node.get_multiplier(node_idx) for node_idx in node.get_children()]
            prob_sum = sum(prob)
            if abs(prob_sum - 1.0) > 1e-9:
                raise ValueError(
                    f"Multipliers for children of node {node.get_idx()} must "
                    f"sum to 1 to be used as sampling probabilities, got {prob_sum}"
                )
            sampled_idx = rng.choice(node.get_children(), p=prob)
            node = self.nodes[sampled_idx]
            path.append(node.get_idx())

            if stage < self.num_stages - 1:
                assert isinstance(node, INodeRoot)
                self._run_forward(node)
        assert not isinstance(node, INodeRoot)
        assert isinstance(node, INodeLeaf)
        up_message = node.get_up_message()  # solve leaf node
        return up_message.get_objective()

    def _run_forward(self, node: INodeRoot) -> float:
        node.reset()
        status, dn_message = node.run_step(None)
        self._last_dn_messages[node.get_idx()] = dn_message

        for child_id in node.get_children():
            child = self.nodes[child_id]
            assert isinstance(child, INodeLeaf)
            child.pass_dn_message(dn_message)
        return dn_message.get_objective()

    def _run_backwards(self) -> None:
        for stage in range(self.num_stages - 1, 0, -1):
            self._run_backward(stage)

    def _run_backward(self, stage: int) -> Dict[NodeIdx, UpMessage]:
        assert stage > 0
        up_messages = {}
        for child_id in self.stages[stage]:
            child = self.nodes[child_id]
            assert isinstance(child, INodeLeaf)
            up_message = child.get_up_message()
            cut_dn = up_message.get_cut()
            assert cut_dn is not None
            if isinstance(cut_dn, OptimalityCut):
                self.logger.log_sub_problem(
                    child.get_idx(), "Optimality", cut_dn.coeffs, cut_dn.rhs
                )
            if isinstance(cut_dn, FeasibilityCut):
                self.logger.log_sub_problem(
                    child.get_idx(), "Feasibility", cut_dn.coeffs, cut_dn.rhs
                )
            up_messages[child_id] = up_message

        for node_idx in self.stages[stage - 1]:
            node = self.nodes[node_idx]
            assert isinstance(node, INodeRoot)
            node.add_cuts(up_messages)

        return up_messages

    def _save(self) -> None:
        for node in self.nodes.values():
            node.save(self.filedir)
        self._save_simulation()
        self._save_stopping()

    def _save_stopping(self) -> None:
        """Write why the run stopped, and the rules it was stopping on."""
        (self.filedir / STOPPING_FILE).write_text(
            json.dumps(
                {
                    "reason": self.stop_reason,
                    "message": self.stop_message,
                    "iterations": len(self.bound_history),
                    "gap_tolerance": self.gap_tolerance,
                    "stable_tolerance": self.stable_tolerance,
                    "stall_iterations": self.stall_iterations,
                    "stall_tolerance": self.stall_tolerance,
                },
                indent=2,
            )
        )

    def _save_simulation(self) -> None:
        """Write the convergence tests to simulation.csv.

        Separate from a node's bm.csv because it belongs to the run rather
        than to any one node, and because it is sampled every
        sample_frequency iterations rather than every iteration.
        """
        pd.DataFrame(
            [round.__dict__ for round in self.simulation_rounds],
            columns=[
                "iteration",
                "bound",
                "mean",
                "lower",
                "upper",
                "sample_size",
                "confidence_level",
            ],
        ).to_csv(self.filedir / SIMULATION_FILE, index=False)

        # The draws behind the last round, for the empirical distribution
        # the interval only summarizes.
        pd.DataFrame({"objective": self.simulation_samples}).to_csv(
            self.filedir / SIMULATION_SAMPLES_FILE, index=False
        )


def _default(value, default):
    return default if value is None else value

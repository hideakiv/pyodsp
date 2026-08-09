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


from pyodsp.alg.params import (
    SDDP_REL_TOLERANCE,
    SDDP_IMPROVE_TOLERANCE,
    SDDP_SEED,
)


SIMULATION_FILE = "simulation.csv"
SIMULATION_SAMPLES_FILE = "simulation_samples.csv"


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
    ) -> None:
        self.num_stages = len(nodes)
        self._verify_nodes(nodes)
        self.logger = logger
        self.filedir = filedir
        self.max_iteration = max_iteration
        self.sample_frequency = sample_frequency
        self.sample_size = sample_size
        self.confidence_level = confidence_level
        create_directory(self.filedir)

        self._last_dn_messages: Dict[NodeIdx, DnMessage] = {}

        self.prev_samples = None
        # One entry per convergence test. SDDP's bound is deterministic but
        # the other side is estimated by simulation, so what it has against
        # the bound is an interval rather than an incumbent — recorded here
        # so a finished run can report and plot it, not just log it.
        self.simulation_rounds: List[SimulationRound] = []
        # The individual draws behind the most recent round. Only the last
        # is kept: it is the converged policy's cost distribution, and
        # every round before it describes a policy that no longer exists.
        self.simulation_samples: List[float] = []

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
        self.logger.log_initialization()
        self._run_init()
        self._run_main()
        self._save()

    def _run_init(self) -> None:
        if self.root is None:
            raise ValueError("Root node not found")
        self.root.set_depth(0)
        for stage in range(self.num_stages - 1):
            self._run_init_forward(stage)

        for stage in range(self.num_stages - 1, 0, -1):
            self._run_init_backward(stage)

    def _run_init_forward(self, stage: int) -> None:
        assert stage < self.num_stages - 1
        for node_idx in self.stages[stage]:
            node = self.nodes[node_idx]
            assert isinstance(node, INodeRoot)
            node.set_logger()

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
        bound = -1e9
        for iteration in range(self.max_iteration):
            bound = self._run_root()
            if iteration % self.sample_frequency == self.sample_frequency - 1:
                if self._termination(bound, iteration):
                    break
            else:
                self._run_forwards(self._iteration_rng(iteration))

            bound = self._run_backwards()

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

    def _termination(self, bound: float, iteration: int = -1) -> bool:
        objectives = self._collect_samples()
        ci_d, ci_u = self._confidence_interval(objectives)
        self.logger.log_info(
            f"lower: {ci_d}, upper: {ci_u}, confidence: {self.confidence_level}"
        )
        self._record_simulation(iteration, objectives, ci_d, ci_u, bound)
        # Always a minimization here — PyomoSolver converts a maximize model
        # on construction — so the sample mean's upper confidence limit is
        # the side that meets the lower bound the algorithm drives up.
        converged = abs(ci_u - bound) / max(abs(ci_u), abs(bound)) < SDDP_REL_TOLERANCE

        if converged:
            self.logger.log_info("SDDP termination with convergence.")
            return True

        no_improve = False
        if self.prev_samples is not None:
            sample_diffs = [
                self.prev_samples[i] - objectives[i] for i in range(len(objectives))
            ]
            all_zero = True
            for sample_diff in sample_diffs:
                if sample_diff > 1e-9:
                    all_zero = False
                    break

            if all_zero:
                no_improve = True
            else:
                _, diff_ci_u = self._confidence_interval(sample_diffs)
                no_improve = diff_ci_u < SDDP_IMPROVE_TOLERANCE

        if no_improve:
            self.logger.log_info("SDDP termination with no improvement.")
            return True

        self.prev_samples = objectives

        return False

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

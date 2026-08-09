"""The multistage stochastic programming front-end."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import pyomo.environ as pyo

from pyodsp.dec.sddp.run import SddpRun
from pyodsp.solver.pyomo_solver import SolverConfig

from .builders import MspBuildContext, MspBuiltProblem, build, build_reference_stage
from .lattice import ScenarioLattice, independent, markov, stage_varying
from .result import MspResult, read_result

DEFAULT_SOLVER = "appsi_highs"


def _normalize_sense(sense) -> bool:
    if sense in (pyo.maximize, "max", "maximize"):
        return True
    if sense in (pyo.minimize, "min", "minimize"):
        return False
    raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")


class MultistageProgram:
    """A multistage stochastic program, solved by SDDP.

    You describe one stage. The same builder runs at every node of the
    lattice, told which stage and which realization it is building, and
    the pipeline handles the rest: replicating the state, matching the
    coupling lists between consecutive stages, the transition
    probabilities, objective-sense conversion, and reading results back.

    The state is the only thing stages share, and it appears twice in
    every stage model — as the value received and the value passed on::

        msp = MultistageProgram("inventory", sense="min")

        @msp.stage(state=["inventory"])
        def stage(m, state, node):
            m.inventory = pyo.Var(bounds=(0, 100))   # passed on
            m.buy = pyo.Var(bounds=(0, 50))
            m.balance = pyo.Constraint(
                expr=m.inventory == state.inventory + m.buy - node["demand"]
            )
            return COST * m.buy

        msp.set_initial_state(inventory=20.0)
        msp.set_realizations(
            [{"name": "low", "probability": 0.5, "demand": 10.0},
             {"name": "high", "probability": 0.5, "demand": 30.0}],
            stages=4,
        )
        result = msp.solve()

    `state.inventory` is what the previous stage left; `m.inventory` is
    what this one leaves. At stage 0 there is no previous stage, so
    `state.inventory` is the initial condition — a number rather than a
    variable, which is why one builder covers every stage.

    Args:
        name: Names the run, its output directory and its models.
        sense: 'min' or 'max', in your own objective's terms.
        solver: Solver name for the stage problems.
        solver_options: Extra keyword arguments passed to that solver.
        output_dir: Where per-node output is written. Defaults to
            output/<name>.
        max_iteration: Iteration cap for SDDP.
        sample_frequency: How often to stop and test convergence by
            simulation, in iterations.
        sample_size: How many scenario paths each of those tests draws.
        confidence_level: The confidence level of the interval that test
            compares against the bound.
        stage_bound: A bound on any single stage's cost-to-go, in your
            own units — a lower bound when minimizing. Every stage needs
            one before its parent can price it; without it the first
            master is unbounded.
        log_level: Logging level for the algorithm's own output.
        validate: Whether to run the structural checks on the models the
            builder produces.
        mpi: Run the Monte Carlo simulation across MPI ranks. Every rank
            builds the identical lattice — unlike the two-stage MPI
            runners, which split distinct nodes across ranks — and rank 0
            drives the iteration while the others help evaluate the
            sample. Launch with `mpiexec -n <ranks> python ...`; results
            are authoritative on rank 0, which is the only rank that
            writes output.
    """

    def __init__(
        self,
        name: str = "msp",
        *,
        sense: str = "min",
        solver: str = DEFAULT_SOLVER,
        solver_options: Mapping[str, Any] | None = None,
        output_dir: str | Path | None = None,
        max_iteration: int = 1000,
        sample_frequency: int = 10,
        sample_size: int = 100,
        confidence_level: float = 0.95,
        stage_bound: float | None = None,
        log_level: int = logging.INFO,
        validate: bool = True,
        mpi: bool = False,
    ) -> None:
        self.name = name
        self.is_maximize = _normalize_sense(sense)
        self.solver_config = SolverConfig(
            solver_name=solver, kwargs=dict(solver_options or {})
        )
        self.output_dir = Path(output_dir) if output_dir else Path("output") / name
        self.max_iteration = max_iteration
        self.sample_frequency = sample_frequency
        self.sample_size = sample_size
        self.confidence_level = confidence_level
        self.stage_bound = stage_bound
        self.log_level = log_level
        self.validate = validate
        self.mpi = mpi

        self._stage_fn: Callable | None = None
        self._state_names: List[str] | None = None
        self._lattice: ScenarioLattice | None = None
        self._initial_state: Dict[str, Any] = {}
        self._built: MspBuiltProblem | None = None

    # -- declaration -------------------------------------------------------

    def stage(self, fn: Callable | None = None, *, state: Sequence[str] | None = None):
        """Register the stage builder.

        The builder takes an empty ConcreteModel, the state the stage
        received, and the LatticeNode it is building; it declares that
        stage's variables and constraints and returns its own cost
        expression, excluding the cost-to-go.

        Args:
            state: Names of the state variables — required. Unlike a
                two-stage program, a stage model holds far more than the
                state, so there is nothing sensible to infer: the builder
                must say which of its variables carry over.
        """

        def decorate(func: Callable) -> Callable:
            if state is None:
                raise ValueError(
                    "stage(state=[...]) is required: a stage model holds its "
                    "own decisions as well as the state, so the state cannot "
                    "be inferred from the variables it declares."
                )
            if not list(state):
                raise ValueError(
                    "state=[] leaves the stages uncoupled, which is a series "
                    "of unrelated problems rather than a multistage one."
                )
            self._stage_fn = func
            self._state_names = list(state)
            self._built = None
            return func

        return decorate if fn is None else decorate(fn)

    def set_initial_state(self, **values: Any) -> "MultistageProgram":
        """The state the first stage starts from.

        Given as numbers: nothing precedes stage 0, so what it receives is
        a condition rather than a decision.
        """
        self._initial_state = dict(values)
        self._built = None
        return self

    def set_realizations(
        self,
        realizations,
        *,
        stages: int,
        normalize: bool = False,
        first_stage_data=None,
    ) -> "MultistageProgram":
        """Stage-wise independent uncertainty.

        The same realizations are drawn at every stage after the first,
        with the same probabilities regardless of what came before.

        Args:
            realizations: In any form as_scenario_set accepts.
            stages: The horizon, counting the first stage.
            normalize: Rescale probabilities to sum to one.
            first_stage_data: Known values for the first stage, which
                observes nothing but may still carry data the builder
                needs.
        """
        self._lattice = independent(
            realizations,
            stages,
            normalize=normalize,
            first_stage_data=first_stage_data,
        )
        self._built = None
        return self

    def set_stage_realizations(
        self, per_stage, *, transitions=None, first_stage_data=None
    ) -> "MultistageProgram":
        """Uncertainty that changes shape over time.

        One collection of realizations per stage after the first, so the
        distribution — and the number of realizations — may differ at
        every stage.

        Args:
            per_stage: The realizations of each stage after the first.
            transitions: One matrix per transition. Left out, each stage
                is drawn independently of the last with its own
                probabilities.
            first_stage_data: Known values for the first stage, which
                observes nothing but may still carry data the builder
                needs.
        """
        self._lattice = stage_varying(
            per_stage,
            transitions=transitions,
            first_stage_data=first_stage_data,
        )
        self._built = None
        return self

    def set_markov_realizations(
        self,
        realizations,
        *,
        stages: int,
        transition_matrix,
        initial_distribution=None,
        first_stage_data=None,
    ) -> "MultistageProgram":
        """Uncertainty that depends on the previous stage's realization."""
        self._lattice = markov(
            realizations,
            stages,
            transition_matrix,
            initial_distribution=initial_distribution,
            first_stage_data=first_stage_data,
        )
        self._built = None
        return self

    def set_lattice(self, lattice: ScenarioLattice) -> "MultistageProgram":
        """Describe the stages and transitions directly."""
        self._lattice = lattice
        self._built = None
        return self

    # -- pipeline ----------------------------------------------------------

    def _context(self, specs=None) -> MspBuildContext:
        if self._stage_fn is None:
            raise ValueError("No stage builder. Decorate one with @msp.stage.")
        if self._lattice is None:
            raise ValueError(
                "No scenario structure. Call set_realizations(...), "
                "set_markov_realizations(...) or set_lattice(...)."
            )
        return MspBuildContext(
            name=self.name,
            stage_fn=self._stage_fn,
            state_names=list(self._state_names or []),
            lattice=self._lattice,
            initial_state=self._initial_state,
            solver_config=self.solver_config,
            is_maximize=self.is_maximize,
            log_level=self.log_level,
            stage_bound=self.stage_bound,
            validate=self.validate,
            purgeable=self.is_root_rank,
            specs=list(specs or []),
        )

    def build(self) -> MspBuiltProblem:
        """Construct the node lattice without running anything."""
        if self._built is not None:
            return self._built

        # The state's shape is learned from stage 0, then every later
        # stage is built against it.
        _, _, specs = build_reference_stage(self._context())
        self._built = build(self._context(specs))
        return self._built

    @property
    def rank(self) -> int:
        """This process's MPI rank, or 0 when not running under MPI."""
        if not self.mpi:
            return 0
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_rank()

    @property
    def is_root_rank(self) -> bool:
        """Whether this process drives the iteration and owns the results.

        Always true without MPI. Under MPI it also decides whether this
        rank's masters purge cuts: only rank 0 may, since a replica that
        aged cuts out on its own schedule would stop matching the trial
        points rank 0 is solving (see LatticeMpi).
        """
        return self.rank == 0

    def _runner(self):
        if not self.mpi:
            return SddpRun
        from pyodsp.dec.sddp.run_mpi import SddpRunMpi

        return SddpRunMpi

    def solve(self, *, max_iteration: int | None = None) -> MspResult:
        """Build if needed, run SDDP, and read the results back.

        Under MPI every rank runs this and every rank returns a result,
        but only rank 0's carries the answer — check `result.is_root_rank`
        before reporting.
        """
        built = self.build()
        self._runner()(
            built.nodes,
            self.output_dir,
            level=self.log_level,
            max_iteration=max_iteration or self.max_iteration,
            sample_frequency=self.sample_frequency,
            sample_size=self.sample_size,
            confidence_level=self.confidence_level,
        ).run()
        return read_result(self, built)

    # -- introspection -----------------------------------------------------

    @property
    def lattice(self) -> ScenarioLattice:
        if self._lattice is None:
            raise ValueError("No scenario structure set")
        return self._lattice

    def plot_scenario_lattice(self, path=None, *, theme: str = "light"):
        """Draw the scenario structure, without solving anything.

        Worth a look before committing to a run: the lattice is where a
        multistage model's size comes from.
        """
        from .viz import plot_scenario_lattice

        return plot_scenario_lattice(self.lattice, path, name=self.name, theme=theme)

    @property
    def state_labels(self) -> List[str]:
        return self.build().labels

    def describe(self) -> str:
        """A plain-language summary of what the pipeline decided."""
        built = self.build()
        lattice = self.lattice
        sizes = [lattice.stage_size(s) for s in range(lattice.num_stages)]
        return "\n".join(
            [
                f"Multistage program {self.name!r}",
                f"  sense           : {'maximize' if self.is_maximize else 'minimize'}",
                f"  stages          : {lattice.num_stages}",
                f"  nodes per stage : {sizes}",
                f"  state variables : {len(built.labels)} "
                f"({', '.join(built.labels[:6])}"
                + ("...)" if len(built.labels) > 6 else ")"),
                f"  initial state   : {self._initial_state}",
                f"  solver          : {self.solver_config.solver_name}",
                f"  output          : {self.output_dir}",
            ]
        )

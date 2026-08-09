"""The two-stage stochastic programming front-end."""

import logging
import warnings
from pathlib import Path
from typing import Any, Callable, List, Mapping, Sequence

import pyomo.environ as pyo

from pyodsp.dec.bd.run import BdRun
from pyodsp.dec.bdsc.run import BdScRun
from pyodsp.dec.dd.run import DdRun
from pyodsp.alg.risk import Expectation, RiskMeasure
from pyodsp.dec.utils import create_directory
from pyodsp.solver.pyomo_solver import SolverConfig

from .builders import (
    BUILDERS,
    BuildContext,
    BuiltProblem,
    build_first_stage_model,
    scan_integers,
)
from .analysis import MEASURES
from .result import SpResult, read_result
from ..scenario import Scenario, ScenarioSet, as_scenario_set
from ..state import resolve_state_specs

DEFAULT_SOLVER = "appsi_highs"
# BDSC prices its columns with a quadratic trust region, so its inner
# master needs a solver that handles one.
DEFAULT_CUT_MASTER_SOLVER = "ipopt"

METHODS = ("bd", "bdsc", "dd", "de")
INTEGER_RECOURSE_POLICIES = ("bdsc", "relax")

RUNNERS = {"bd": BdRun, "bdsc": BdScRun, "dd": DdRun}


def _normalize_sense(sense) -> bool:
    """True for maximize."""
    if sense in (pyo.maximize, "max", "maximize"):
        return True
    if sense in (pyo.minimize, "min", "minimize"):
        return False
    raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")


class StochasticProgram:
    """A two-stage stochastic program, solved by decomposition.

    You describe the problem twice — once for the first stage, once for
    one representative scenario's recourse — and this class handles the
    decomposition: which algorithm suits the model, how the stages couple,
    objective-sense conversion, subproblem bounds, and reading results
    back in your own units::

        sp = StochasticProgram("farmer", sense="max")

        @sp.first_stage
        def first_stage(m):
            m.acres = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
            m.land = pyo.Constraint(expr=sum(m.acres[c] for c in CROPS) <= 500)
            return -sum(COST[c] * m.acres[c] for c in CROPS)

        @sp.recourse
        def recourse(m, state, scenario):
            m.sold = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
            m.balance = pyo.Constraint(
                CROPS,
                rule=lambda m, c: m.sold[c] == scenario["yield"][c] * state.acres[c],
            )
            return sum(PRICE[c] * m.sold[c] for c in CROPS)

        sp.set_scenarios(...)
        result = sp.solve()

    Args:
        name: Names the run, its output directory and its models.
        sense: 'min' or 'max', in your own objective's terms. The
            algorithms only accept minimize problems; a maximize program
            is converted internally and converted back on the way out.
        method: 'bd' for Benders decomposition, 'dd' for dual
            decomposition, 'bdsc' to force Benders with scaled cuts, or
            'de' to skip decomposition entirely and hand the solver the
            deterministic equivalent (the extensive form). See `resolve_method` for what 'bd'
            does when the recourse turns out to be integral.
        integer_recourse: What 'bd' should do about integer recourse
            variables — 'bdsc' to switch to Benders with scaled cuts,
            which handles them exactly, or 'relax' to solve the LP
            relaxation of the second stage.
        solver: Solver name for the stage problems.
        solver_options: Extra keyword arguments passed to that solver.
        cut_master_solver: Solver for BDSC's inner column-generation
            master, which is quadratic.
        output_dir: Where per-node output is written. Defaults to
            output/<name>.
        max_iteration: Iteration cap for the algorithm.
        recourse_bound: A known bound on any single scenario's recourse
            objective, in your own units — an upper bound when
            maximizing, a lower bound when minimizing. Left unset, one is
            computed per scenario.
        auto_bound: Whether to compute that bound when it is not given.
            Costs one extra solve per scenario at build time.
        heuristic: For dual decomposition, whether to recover a feasible
            solution with a MIP heuristic at the end. Without it you get
            a bound and no primal solution.
        validate: Whether to run the structural checks on the models the
            builders produce — that no scenario overwrote a state
            variable, that no integer recourse variable reaches Benders,
            and that BDSC and DD get the complete state vector. Each
            costs a pass over a built model, and the last two need a
            probe model of their own, so turning them off is worth it on
            a large problem you have already run once. Nothing about the
            requirements changes; they simply stop being enforced, and a
            violation then produces a confidently wrong answer rather
            than an error.
        risk: How the scenarios are valued. Expectation by default —
            minimize the average cost, which is what a stochastic program
            means unqualified. Pass CVaR(alpha=..., weight=...) to weight
            the bad tail instead, which changes the problem rather than
            the way it is solved. Only method='bd' and method='de' support
            one; see `resolve_method`.
        log_level: Logging level for the algorithm's own output.
    """

    def __init__(
        self,
        name: str = "sp",
        *,
        sense: str = "min",
        method: str = "bd",
        integer_recourse: str = "bdsc",
        solver: str = DEFAULT_SOLVER,
        solver_options: Mapping[str, Any] | None = None,
        cut_master_solver: str = DEFAULT_CUT_MASTER_SOLVER,
        output_dir: str | Path | None = None,
        max_iteration: int = 1000,
        recourse_bound: float | None = None,
        auto_bound: bool = True,
        heuristic: bool = True,
        validate: bool = True,
        risk: RiskMeasure | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if integer_recourse not in INTEGER_RECOURSE_POLICIES:
            raise ValueError(
                f"integer_recourse must be one of {INTEGER_RECOURSE_POLICIES}, "
                f"got {integer_recourse!r}"
            )

        self.name = name
        self.is_maximize = _normalize_sense(sense)
        self.method = method
        self.integer_recourse = integer_recourse
        self.solver_config = SolverConfig(
            solver_name=solver, kwargs=dict(solver_options or {})
        )
        self.cut_master_config = SolverConfig(solver_name=cut_master_solver)
        self.output_dir = Path(output_dir) if output_dir else Path("output") / name
        self.max_iteration = max_iteration
        self.recourse_bound = recourse_bound
        self.auto_bound = auto_bound
        self.heuristic = heuristic
        self.validate = validate
        self.risk: RiskMeasure = risk or Expectation()
        self.log_level = log_level

        self._first_stage_fn: Callable | None = None
        self._state_names: Sequence[str] | None = None
        self._recourse_fn: Callable | None = None
        self._scenarios: ScenarioSet | None = None
        self._pending: List[Scenario] = []

        self._built: BuiltProblem | None = None
        self._resolved_method: str | None = None
        self._integers = None

    # -- declaration -------------------------------------------------------

    def first_stage(self, fn: Callable | None = None, *, state=None):
        """Register the first-stage builder.

        The builder takes an empty ConcreteModel, declares the here-and-now
        variables and constraints on it, and returns the first-stage
        objective expression.

        Args:
            state: Names of the variables the recourse problems see. All
                first-stage variables by default, which is what a
                two-stage program normally wants.
        """

        def decorate(func: Callable) -> Callable:
            self._first_stage_fn = func
            self._state_names = list(state) if state is not None else None
            self._invalidate()
            return func

        return decorate if fn is None else decorate(fn)

    def recourse(self, fn: Callable | None = None):
        """Register the recourse builder.

        The builder takes an empty ConcreteModel, a `state` giving that
        model's own copy of each first-stage variable, and the Scenario;
        it returns the recourse objective expression. It is called once
        per scenario.
        """

        def decorate(func: Callable) -> Callable:
            self._recourse_fn = func
            self._invalidate()
            return func

        return decorate if fn is None else decorate(fn)

    def set_scenarios(
        self, scenarios, *, normalize: bool = False
    ) -> "StochasticProgram":
        """Set the scenarios from data.

        Accepts a ScenarioSet, a pandas DataFrame, a mapping of name to
        data, or an iterable of Scenario or of dicts.
        """
        self._scenarios = as_scenario_set(scenarios, normalize=normalize)
        self._pending = []
        self._invalidate()
        return self

    def add_scenario(
        self,
        name: str,
        data: Mapping[str, Any] | None = None,
        *,
        probability: float | None = None,
        **extra: Any,
    ) -> "StochasticProgram":
        """Append one scenario.

        Args:
            name: Its name.
            data: Its realized data. Merged with any extra keyword
                arguments, so both spellings work.
            probability: Its probability. When left unset for every
                scenario, they end up equally likely.
        """
        merged = dict(data or {})
        merged.update(extra)
        # Probability is filled in at build time, once the count is known.
        self._pending.append(
            Scenario(name, float("nan") if probability is None else probability, merged)
        )
        self._scenarios = None
        self._invalidate()
        return self

    def _invalidate(self) -> None:
        self._built = None
        self._resolved_method = None
        self._integers = None

    def _resolve_scenarios(self) -> ScenarioSet:
        if self._scenarios is not None:
            return self._scenarios
        if not self._pending:
            raise ValueError(
                "No scenarios. Call set_scenarios(...) or add_scenario(...)."
            )
        unset = [s for s in self._pending if s.probability != s.probability]
        if unset and len(unset) != len(self._pending):
            raise ValueError(
                "Either every add_scenario call gives a probability or none "
                "does; a partial set has no sensible completion."
            )
        if unset:
            weight = 1.0 / len(self._pending)
            resolved = [Scenario(s.name, weight, s.data) for s in self._pending]
        else:
            resolved = list(self._pending)
        self._scenarios = ScenarioSet(resolved)
        return self._scenarios

    # -- pipeline ----------------------------------------------------------

    def _context(self, *, relax_recourse: bool) -> BuildContext:
        if self._first_stage_fn is None:
            raise ValueError(
                "No first-stage builder. Decorate one with @sp.first_stage."
            )
        if self._recourse_fn is None:
            raise ValueError("No recourse builder. Decorate one with @sp.recourse.")

        scenarios = self._resolve_scenarios()
        reference_model, reference_expr = build_first_stage_model(
            self._first_stage_fn, f"{self.name}_first_stage"
        )
        specs = resolve_state_specs(reference_model, self._state_names)

        return BuildContext(
            name=self.name,
            first_stage_fn=self._first_stage_fn,
            recourse_fn=self._recourse_fn,
            specs=specs,
            reference_model=reference_model,
            reference_expr=reference_expr,
            scenarios=scenarios,
            solver_config=self.solver_config,
            cut_master_config=self.cut_master_config,
            is_maximize=self.is_maximize,
            max_iteration=self.max_iteration,
            log_level=self.log_level,
            recourse_bound=self.recourse_bound,
            auto_bound=self.auto_bound,
            relax_recourse=relax_recourse,
            heuristic=self.heuristic,
            validate=self.validate,
            risk=self.risk,
        )

    def resolve_method(self, ctx: BuildContext) -> tuple[str, bool]:
        """Pick the algorithm actually used, and whether to relax.

        'dd', 'bdsc' and 'de' are taken at face value; 'dd' only warns
        about what integrality costs it. 'bd' is the one that adapts: Benders cuts are
        built from LP duals, which an integer second stage does not have,
        so an integer recourse either moves the problem to Benders with
        scaled cuts or has its integrality relaxed away — never silently
        stays on plain Benders, which would return a wrong answer.
        """
        # A precondition on the method as asked for, so it is checked
        # before anything that might return early.
        if not self.risk.is_risk_neutral:
            self._require_risk_capable_method()

        # Probing costs a built model, so it only happens where the answer
        # is used: 'bdsc' and 'de' are taken as given — the deterministic
        # equivalent hands integrality straight to the solver — and for
        # 'dd' the scan feeds a warning that validate=False has opted out of.
        if self.method in ("bdsc", "de") or (self.method == "dd" and not self.validate):
            return self.method, False

        scan = scan_integers(ctx)
        self._integers = scan

        if self.method == "dd":
            if scan.any:
                found = sorted(set(scan.first_stage + scan.recourse))
                shown = ", ".join(found[:5]) + ("..." if len(found) > 5 else "")
                warnings.warn(
                    f"Dual decomposition on a problem with integer variables "
                    f"({shown}) solves the Lagrangian dual. For a non-convex "
                    "problem that sits strictly below the true optimum, so "
                    "the reported bound is a lower bound only, the incumbent "
                    "comes from a heuristic, and the gap between them need "
                    "not close. Use method='bd' if you need a proven optimum.",
                    UserWarning,
                    stacklevel=3,
                )
            return "dd", False

        if not scan.recourse:
            return "bd", False

        found = sorted(set(scan.recourse))
        shown = ", ".join(found[:5]) + ("..." if len(found) > 5 else "")
        if self.integer_recourse == "bdsc" and not self.risk.is_risk_neutral:
            raise ValueError(
                f"The recourse problem has integer variables ({shown}), which "
                "would normally send this to Benders with scaled cuts — but "
                f"that cannot carry a risk measure ({self.risk.describe()}), "
                "because it aggregates the scenarios into a single cut and a "
                "risk measure prices the spread between them. Use "
                "integer_recourse='relax', or method='de' to solve the "
                "deterministic equivalent, which handles both."
            )
        if self.integer_recourse == "bdsc":
            warnings.warn(
                f"The recourse problem has integer variables ({shown}), whose "
                "LP duals Benders would need and which do not exist. Switching "
                "to Benders with scaled cuts, which handles them exactly. Pass "
                "integer_recourse='relax' to solve the LP relaxation instead, "
                f"or method='dd'. BDSC needs a quadratic-capable solver for "
                f"its inner master (currently "
                f"{self.cut_master_config.solver_name!r}).",
                UserWarning,
                stacklevel=3,
            )
            return "bdsc", False

        warnings.warn(
            f"Relaxing the integrality of the recourse variables ({shown}) so "
            "Benders applies. The result bounds the true optimum rather than "
            "attaining it, and the second-stage solution may be fractional. "
            "Pass integer_recourse='bdsc' to solve the integer problem "
            "exactly.",
            UserWarning,
            stacklevel=3,
        )
        return "bd", True

    def _require_risk_capable_method(self) -> None:
        """Refuse a risk measure where the algorithm cannot represent it.

        CVaR is priced on the *spread* across scenarios, so the master has
        to see each scenario's cost separately. Benders does — one theta
        per scenario. BDSC aggregates them into a single cut, and dual
        decomposition puts a whole copy of the problem in every scenario
        with nothing comparing them, so neither can state the tail.
        """
        if self.method in ("bd", "de"):
            return
        raise ValueError(
            f"method={self.method!r} cannot carry a risk measure "
            f"({self.risk.describe()}). A risk measure prices the spread "
            "across scenarios, which needs each scenario's cost visible to "
            "the master separately — Benders keeps one theta per scenario, "
            "but BDSC aggregates them into a single cut and dual "
            "decomposition never compares them. Use method='bd', or "
            "method='de' to solve the deterministic equivalent."
        )

    def build(self) -> BuiltProblem:
        """Construct the node graph without running anything.

        Useful on its own to inspect what the pipeline decided — which
        algorithm, which state vector — before committing to a solve.
        """
        if self._built is not None:
            return self._built

        # The method decision needs a context to probe, and a relaxing
        # build needs the decision, so the context is built twice. Only
        # the second one is kept and wired into nodes.
        method, relax = self.resolve_method(self._context(relax_recourse=False))
        ctx = self._context(relax_recourse=relax)
        self._resolved_method = method
        self._built = BUILDERS[method](ctx)
        return self._built

    def solve(self, *, max_iteration: int | None = None) -> SpResult:
        """Build if needed, run the algorithm, and read the results back.

        Returns an SpResult carrying the first-stage decision, the
        per-scenario outcome and the convergence history, all in your own
        objective's units.
        """
        built = self.build()
        assert self._resolved_method is not None

        if self._resolved_method == "de":
            self._solve_deterministic_equivalent(built)
        else:
            runner = RUNNERS[self._resolved_method](
                built.nodes,
                self.output_dir,
                level=self.log_level,
                max_iteration=max_iteration or self.max_iteration,
            )
            runner.run()

        return read_result(self, built)

    def _solve_deterministic_equivalent(self, built: BuiltProblem) -> None:
        """Hand the deterministic equivalent to the solver in one piece.

        There is no master, no subproblems and no iteration here, so none
        of the Run classes apply — but the output directory is still
        written, so a run is inspectable the same way.
        """
        solver = built.root_solver
        assert solver is not None
        solver.solve()
        if not solver.is_optimal():
            raise RuntimeError(
                "The deterministic equivalent did not solve to optimality "
                f"({solver._results.solver.termination_condition}). With no "
                "decomposition involved this is the model itself — check it "
                "is bounded and feasible."
            )
        create_directory(self.output_dir)
        solver.save(self.output_dir)

    def analyze(
        self,
        *,
        measures: Sequence[str] = MEASURES,
        mean_scenario_data: Mapping[str, Any] | None = None,
    ):
        """What the uncertainty is worth: EVPI and VSS.

        Solves this program, then the reference problems the two measures
        compare it against — one solve per scenario for EVPI, two more
        for VSS. See pyodsp.model.sp.analysis for what each means.

            analysis = sp.analyze()
            print(analysis.summary())
            analysis.plot()

        Args:
            measures: Which of 'evpi' and 'vss' to compute; each skips
                the other's solves when left out.
            mean_scenario_data: Entries to use for the expected-value
                problem instead of the averaged ones — the way out for
                data that has no meaningful mean.
        """
        from .analysis import analyze

        return analyze(self, measures=measures, mean_scenario_data=mean_scenario_data)

    # -- introspection -----------------------------------------------------

    @property
    def scenarios(self) -> ScenarioSet:
        return self._resolve_scenarios()

    @property
    def resolved_method(self) -> str | None:
        """Which algorithm build() settled on, or None before building."""
        return self._resolved_method

    @property
    def state_labels(self) -> List[str]:
        return self.build().labels

    def describe(self) -> str:
        """A plain-language summary of what the pipeline decided."""
        built = self.build()
        lines = [
            f"Stochastic program {self.name!r}",
            f"  sense           : {'maximize' if self.is_maximize else 'minimize'}",
            f"  algorithm       : {self._resolved_method}"
            + (
                ""
                if self._resolved_method == self.method
                else f" (asked for {self.method})"
            ),
            f"  scenarios       : {len(built.leaf_nodes)}",
            f"  state variables : {len(built.labels)} ({', '.join(built.labels[:6])}"
            + ("...)" if len(built.labels) > 6 else ")"),
            f"  risk            : {self.risk.describe()}",
            f"  solver          : {self.solver_config.solver_name}",
            f"  output          : {self.output_dir}",
        ]
        if self._integers is not None and self._integers.any:
            lines.append(
                f"  integer vars    : {len(self._integers.first_stage)} first-stage, "
                f"{len(self._integers.recourse)} recourse"
            )
        if built.relaxed_variables:
            relaxed = sum(len(v) for v in built.relaxed_variables.values())
            lines.append(f"  relaxed         : {relaxed} recourse integer variables")
        return "\n".join(lines)

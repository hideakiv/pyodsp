"""Reading a finished run back in the user's own objective units.

The algorithms work in the minimize form PyomoSolver converted the
models to; each solver remembers whether it flipped, so the conversion
back happens through `to_user_units` rather than by tracking a sign
here. Saved trajectories are already written in the user's units by
BundleMethod.save, so nothing on disk needs correcting.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pyomo.environ as pyo

from pyodsp.alg.risk import Expectation, value_of_sample

STATE_CONSISTENCY_TOL = 1e-6


@dataclass
class ScenarioOutcome:
    """What one scenario contributed.

    Attributes:
        name: The scenario's name.
        probability: Its probability.
        objective: Its model's own objective value, in your units. Under
            Benders (and BDSC) that is the recourse objective alone,
            since the first stage lives in the master. Under dual
            decomposition each scenario carries a complete, probability-
            weighted copy of the problem, so it is that scenario's
            weighted share of the total instead.
        variables: Every variable in the scenario model, by name.
    """

    name: str
    probability: float
    objective: float | None
    variables: Dict[str, float] = field(default_factory=dict)


@dataclass
class SpResult:
    """The outcome of StochasticProgram.solve.

    Attributes:
        name: The program's name.
        method: The algorithm that actually ran.
        is_maximize: Whether the objective was maximized.
        objective: The value of the returned solution under the program's
            risk measure — what was actually optimized — in your units, or
            None if no solution was recovered. Equal to
            `expected_objective` when the program is risk-neutral.
        expected_objective: The probability-weighted average objective,
            whatever the risk measure. Under CVaR this is what the risk
            aversion cost you.
        risk: How the scenarios were valued.
        bound: The algorithm's final bound on the optimum, in your units.
        first_stage: The here-and-now decision, nested by variable name —
            a float for a scalar variable, {index: value} for an indexed
            one.
        scenarios: Per-scenario outcomes, in scenario order.
        history: Per-iteration `bound` and `incumbent`, in your units.
        first_stage_consistent: Whether every scenario agrees on the
            first-stage decision. Always true for Benders, where there is
            one master; informative for dual decomposition, where it
            means non-anticipativity was actually attained.
        output_dir: Where the run's per-node files were written.
    """

    name: str
    method: str
    is_maximize: bool
    objective: float | None
    expected_objective: float | None
    risk: Any
    bound: float | None
    first_stage: Dict[str, Any]
    scenarios: List[ScenarioOutcome]
    history: pd.DataFrame
    labels: List[str]
    first_stage_flat: Dict[str, float]
    first_stage_consistent: bool
    output_dir: Path

    @property
    def gap(self) -> float | None:
        """Relative distance between the incumbent and the bound."""
        if self.objective is None or self.bound is None:
            return None
        scale = max(abs(self.objective), 1e-10)
        return abs(self.objective - self.bound) / scale

    def summary(self) -> str:
        sense = "maximize" if self.is_maximize else "minimize"
        lines = [f"{self.name}: {sense} via {self.method}"]
        if self.risk is not None and not self.risk.is_risk_neutral:
            lines.append(f"  risk      : {self.risk.describe()}")
            lines.append(f"  objective : {_fmt(self.objective)}  (risk-adjusted)")
            lines.append(f"  expected  : {_fmt(self.expected_objective)}")
        else:
            lines.append(f"  objective : {_fmt(self.objective)}")
        lines.append(f"  bound     : {_fmt(self.bound)}")
        if self.gap is not None:
            lines.append(f"  gap       : {self.gap:.4%}")
        lines.append(f"  iterations: {len(self.history)}")
        if not self.first_stage_consistent:
            lines.append("  warning   : scenarios disagree on the first-stage decision")
        lines.append("  first stage:")
        for label, value in self.first_stage_flat.items():
            lines.append(f"    {label} = {_fmt(value)}")
        lines.append("  scenarios:")
        for outcome in self.scenarios:
            lines.append(
                f"    {outcome.name} (p={outcome.probability:.4g}): "
                f"{_fmt(outcome.objective)}"
            )
        return "\n".join(lines)

    def first_stage_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "variable": list(self.first_stage_flat),
                "value": list(self.first_stage_flat.values()),
            }
        )

    def scenario_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "scenario": [s.name for s in self.scenarios],
                "probability": [s.probability for s in self.scenarios],
                "objective": [s.objective for s in self.scenarios],
            }
        )

    def plot(self, directory: str | Path | None = None) -> List[Path]:
        """Write every applicable chart. Requires matplotlib."""
        from .viz import plot_all

        return plot_all(self, directory or self.output_dir)

    def __repr__(self) -> str:
        return (
            f"SpResult(name={self.name!r}, method={self.method!r}, "
            f"objective={self.objective!r}, scenarios={len(self.scenarios)})"
        )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def read_result(program, built) -> SpResult:
    """Assemble an SpResult from the solved problem."""
    if built.method == "de":
        return _read_deterministic_equivalent(program, built)

    scenarios = program.scenarios

    outcomes: List[ScenarioOutcome] = []
    for scenario in scenarios:
        solver = built.leaf_solvers[scenario.name]
        outcomes.append(
            ScenarioOutcome(
                name=scenario.name,
                probability=scenario.probability,
                objective=solver.to_user_units(solver.get_original_objective_value()),
                variables=solver.get_variable_values(),
            )
        )

    first_stage, flat, consistent = _read_first_stage(program, built)
    expectation = _total_objective(built, outcomes)
    objective, _ = _risk_adjusted(
        program, built, outcomes, _first_stage_objective(built)
    )
    if objective is None:
        objective = expectation
    bound = _final_bound(built)
    history = _read_history(program.output_dir)

    return SpResult(
        name=program.name,
        method=built.method,
        is_maximize=program.is_maximize,
        objective=objective,
        expected_objective=expectation,
        risk=getattr(program, "risk", None),
        bound=bound,
        first_stage=first_stage,
        scenarios=outcomes,
        history=history,
        labels=list(built.labels),
        first_stage_flat=flat,
        first_stage_consistent=consistent,
        output_dir=Path(program.output_dir),
    )


def _risk_adjusted(program, built, outcomes, first_stage: float | None):
    """The value of this solution under the program's risk measure.

    Computed from the same per-scenario costs the master priced, in the
    minimize convention the risk measure is stated in, and converted back.
    """
    risk = getattr(program, "risk", None) or Expectation()
    if first_stage is None or any(o.objective is None for o in outcomes):
        return None, None

    sign = -1.0 if program.is_maximize else 1.0
    recourse = [sign * outcome.objective for outcome in outcomes]
    probabilities = [outcome.probability for outcome in outcomes]

    expectation = (
        first_stage
        + sign * sum(p * value for p, value in zip(probabilities, recourse)) * 1.0
    )
    risked = first_stage + sign * value_of_sample(risk, recourse, probabilities)
    return risked, expectation


def _block_variable_values(block) -> Dict[str, float]:
    """Every variable on one scenario block, keyed by its local name."""
    solution = {}
    for component in block.component_objects(pyo.Var, active=True):
        for index in component:
            name = (
                component.local_name
                if index is None
                else f"{component.local_name}_{index}"
            )
            solution[name] = component[index].value
    return solution


def _read_deterministic_equivalent(program, built) -> SpResult:
    """Read back the deterministic equivalent.

    A different shape from the decomposed runs: one model, one solve, no
    iteration. Each scenario's contribution comes from evaluating the
    expression its builder returned — which was never negated, so it is
    already in the user's units whatever the sense — and the bound equals
    the objective, since the solver proved optimality on the whole
    problem rather than approaching it.
    """
    solver = built.root_solver
    assert solver is not None

    outcomes = [
        ScenarioOutcome(
            name=scenario.name,
            probability=scenario.probability,
            objective=pyo.value(built.recourse_exprs[scenario.name]),
            variables=_block_variable_values(built.scenario_blocks[scenario.name]),
        )
        for scenario in program.scenarios
    ]

    # The model's own objective already carries the risk measure, since
    # build_de put it there; the expectation is recomputed alongside it.
    objective = solver.to_user_units(solver.get_original_objective_value())
    first_stage = pyo.value(built.first_stage_expr)
    _, expectation = _risk_adjusted(program, built, outcomes, first_stage)
    values = [var.value for var in solver.get_vars()]
    flat = {
        label: value for label, value in zip(built.labels, values) if value is not None
    }

    return SpResult(
        name=program.name,
        method="de",
        is_maximize=program.is_maximize,
        objective=objective,
        expected_objective=expectation,
        risk=getattr(program, "risk", None),
        bound=objective,
        first_stage=_nest(built, built.labels, values),
        scenarios=outcomes,
        history=pd.DataFrame(columns=["iteration", "bound", "incumbent"]),
        labels=list(built.labels),
        first_stage_flat=flat,
        first_stage_consistent=True,
        output_dir=Path(program.output_dir),
    )


def _read_first_stage(program, built):
    """The here-and-now decision, and whether the scenarios agree on it."""
    master = built.root_solver
    labels = built.labels

    per_scenario = []
    for scenario in program.scenarios:
        solver = built.leaf_solvers[scenario.name]
        per_scenario.append([var.value for var in solver.get_vars()])

    if master is not None:
        values = [var.value for var in master.get_vars()]
        consistent = True
    else:
        # Dual decomposition has no master model: the decision is whatever
        # the scenarios agreed on, which is only meaningful if they did.
        values = per_scenario[0] if per_scenario else []
        consistent = _all_close(per_scenario)

    flat = {label: value for label, value in zip(labels, values) if value is not None}
    return _nest(built, labels, values), flat, consistent


def _all_close(vectors: List[List[float | None]]) -> bool:
    if len(vectors) < 2:
        return True
    first = vectors[0]
    for other in vectors[1:]:
        for a, b in zip(first, other):
            if a is None or b is None:
                return False
            if abs(a - b) > STATE_CONSISTENCY_TOL * max(1.0, abs(a), abs(b)):
                return False
    return True


def _nest(built, labels: List[str], values: List[float | None]) -> Dict[str, Any]:
    """Group the flat state vector back into {component: value-or-dict}."""
    nested: Dict[str, Any] = {}
    for label, value in zip(labels, values):
        if "[" not in label:
            nested[label] = value
            continue
        name, _, index = label.partition("[")
        nested.setdefault(name, {})[index.rstrip("]")] = value
    return nested


def _first_stage_objective(built) -> float | None:
    """The here-and-now cost alone, in the user's units."""
    if built.root_solver is None:
        return None
    return built.root_solver.to_user_units(
        built.root_solver.get_original_objective_value()
    )


def _total_objective(built, outcomes) -> float | None:
    """Expected objective of the returned solution, in the user's units."""
    if any(outcome.objective is None for outcome in outcomes):
        return None

    if built.method == "dd":
        # Each scenario already carries its probability-weighted share of
        # the whole problem, first stage included.
        return sum(outcome.objective for outcome in outcomes)

    assert built.root_solver is not None
    first_stage = built.root_solver.to_user_units(
        built.root_solver.get_original_objective_value()
    )
    if first_stage is None:
        return None
    return first_stage + sum(o.probability * o.objective for o in outcomes)


def _final_bound(built) -> float | None:
    """The master's last bound, already in the user's units."""
    alg_root = getattr(built.root_node, "alg_root", None)
    bm = getattr(alg_root, "bm", None)
    if bm is None:
        return None
    return bm.get_objective_bound()


def _read_history(output_dir: str | Path) -> pd.DataFrame:
    """The master's per-iteration trajectory, already in the user's units."""
    path = Path(output_dir) / "node0" / "bm.csv"
    columns = ["iteration", "bound", "incumbent"]
    if not path.exists():
        return pd.DataFrame(columns=columns)

    frame = pd.read_csv(path, index_col=0)
    missing = [float("nan")] * len(frame)
    # .to_numpy() rather than the Series: the saved index is the algorithm's
    # own, and letting pandas align on it would silently reorder the
    # trajectory if it were ever anything but 0..n-1.
    return pd.DataFrame(
        {
            "iteration": range(len(frame)),
            "bound": frame["obj_bound"].to_numpy() if "obj_bound" in frame else missing,
            "incumbent": frame["obj_val"].to_numpy() if "obj_val" in frame else missing,
        }
    )

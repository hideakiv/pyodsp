"""Reading a finished run back in the user's own objective units.

Everything the algorithms produce is in internal minimize form. A
maximize program was negated on the way in, so every number here is
negated on the way out — including the trajectories already written to
disk, which are corrected in place so the saved artifacts agree with the
returned result.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from pyodsp.solver.pyomo_utils import negate_saved_objective_csv

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
        objective: The expected objective of the returned solution, in
            your units, or None if no solution was recovered.
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
        lines = [
            f"{self.name}: {sense} via {self.method}",
            f"  objective : {_fmt(self.objective)}",
            f"  bound     : {_fmt(self.bound)}",
        ]
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
    """Assemble an SpResult from the solved node graph."""
    sign = -1.0 if program.is_maximize else 1.0
    scenarios = program.scenarios

    _correct_saved_trajectories(program, built)

    outcomes: List[ScenarioOutcome] = []
    for scenario in scenarios:
        solver = built.leaf_solvers[scenario.name]
        objective = solver.get_original_objective_value()
        outcomes.append(
            ScenarioOutcome(
                name=scenario.name,
                probability=scenario.probability,
                objective=None if objective is None else sign * objective,
                variables=solver.get_variable_values(),
            )
        )

    first_stage, flat, consistent = _read_first_stage(program, built)
    objective = _total_objective(built, outcomes, sign)
    bound = _final_bound(built, sign)
    history = _read_history(program.output_dir)

    return SpResult(
        name=program.name,
        method=built.method,
        is_maximize=program.is_maximize,
        objective=objective,
        bound=bound,
        first_stage=first_stage,
        scenarios=outcomes,
        history=history,
        labels=list(built.labels),
        first_stage_flat=flat,
        first_stage_consistent=consistent,
        output_dir=Path(program.output_dir),
    )


def _correct_saved_trajectories(program, built) -> None:
    """Put the saved bm.csv trajectories back into the user's units.

    The graph classes save whatever sense the node's model had, with no
    notion of "this was negated" — so for a maximize program the files on
    disk are left in internal form unless corrected here.
    """
    if not program.is_maximize:
        return
    root = Path(program.output_dir)
    if not root.exists():
        return
    for node_dir in sorted(root.glob("node*")):
        try:
            negate_saved_objective_csv(node_dir)
        except FileNotFoundError:
            # Benders leaves save solutions and timings but no trajectory.
            continue


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


def _total_objective(built, outcomes, sign: float) -> float | None:
    """Expected objective of the returned solution, in the user's units."""
    if any(outcome.objective is None for outcome in outcomes):
        return None

    if built.method == "dd":
        # Each scenario already carries its probability-weighted share of
        # the whole problem, first stage included.
        return sum(outcome.objective for outcome in outcomes)

    assert built.root_solver is not None
    first_stage = built.root_solver.get_original_objective_value()
    if first_stage is None:
        return None
    total = sign * first_stage
    total += sum(o.probability * o.objective for o in outcomes)
    return total


def _final_bound(built, sign: float) -> float | None:
    """The master's last bound, read off the bundle method."""
    alg_root = getattr(built.root_node, "alg_root", None)
    bm = getattr(alg_root, "bm", None)
    bounds = getattr(bm, "obj_bound", None)
    if not bounds:
        return None
    last = bounds[-1]
    return None if last is None else sign * last


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

"""What the uncertainty is worth: EVPI and VSS.

Neither is a problem you solve — each is a comparison between problems,
so both come from running the stochastic program against two reference
points (Birge and Louveaux, ch. 4):

    WS   each scenario solved as if it were known in advance, averaged.
         The cost of a decision maker who sees the future.
    RP   the stochastic program itself.
    EEV  the expected-value problem's decision, evaluated against every
         scenario. The cost of pretending the mean is the truth.

For a minimization WS <= RP <= EEV, and the two gaps are

    EVPI = RP - WS     what perfect information would be worth
    VSS  = EEV - RP    what modelling the uncertainty was worth

Maximization reverses both orderings, so the differences are signed by
the sense and both measures stay non-negative either way.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .mean import mean_scenario
from .result import SpResult
from .scenario import Scenario, ScenarioSet

MEASURES = ("evpi", "vss")


@dataclass
class SpAnalysis:
    """EVPI and VSS, with the solves they came from.

    Attributes:
        name: The program's name.
        is_maximize: Whether the objective was maximized.
        rp: The stochastic program's own result.
        ws: The wait-and-see value, or None if not computed.
        ev: The expected-value problem's result, or None.
        eev: The expected result of using the EV decision, or None. None
            also when that decision turned out infeasible for some
            scenario — see `eev_infeasible`.
        eev_infeasible: True when the EV decision cannot be carried out
            in every scenario, which makes VSS unbounded rather than
            merely large.
        scenario_values: Each scenario's wait-and-see optimum, by name.
        output_dir: Where the supporting runs were written.
    """

    name: str
    is_maximize: bool
    rp: SpResult
    ws: float | None = None
    ev: SpResult | None = None
    eev: float | None = None
    eev_infeasible: bool = False
    scenario_values: Dict[str, float] = field(default_factory=dict)
    output_dir: Path | None = None

    @property
    def sense_multiplier(self) -> float:
        """+1 minimizing, -1 maximizing — what keeps the gaps positive."""
        return -1.0 if self.is_maximize else 1.0

    @property
    def evpi(self) -> float | None:
        """What perfect information would be worth.

        Zero means the same decision is optimal however the uncertainty
        resolves, so knowing the future in advance would buy nothing.
        """
        if self.ws is None or self.rp.objective is None:
            return None
        return (self.rp.objective - self.ws) * self.sense_multiplier

    @property
    def vss(self) -> float | None:
        """What modelling the uncertainty was worth.

        Zero means planning against the mean happens to give the same
        decision as the stochastic program — the uncertainty did not need
        modelling. It is never negative: the stochastic solution is
        optimal over the same feasible set the EV decision came from.
        """
        if self.eev is None or self.rp.objective is None:
            return None
        return (self.eev - self.rp.objective) * self.sense_multiplier

    @property
    def relative_vss(self) -> float | None:
        """VSS as a fraction of the stochastic optimum."""
        return self._relative(self.vss)

    @property
    def relative_evpi(self) -> float | None:
        return self._relative(self.evpi)

    def _relative(self, value: float | None) -> float | None:
        if value is None or not self.rp.objective:
            return None
        return value / abs(self.rp.objective)

    def summary(self) -> str:
        lines = [
            f"{self.name}: value of information",
            f"  WS  (perfect information) : {_fmt(self.ws)}",
            f"  RP  (stochastic program)  : {_fmt(self.rp.objective)}",
            f"  EEV (mean-value decision) : {_fmt(self.eev)}",
        ]
        if self.eev_infeasible:
            lines.append(
                "        the mean-value decision is infeasible in at least "
                "one scenario"
            )
        lines.append("")
        if self.evpi is not None:
            lines.append(
                f"  EVPI = RP - WS   : {_fmt(self.evpi)}" + _percent(self.relative_evpi)
            )
        if self.vss is not None:
            lines.append(
                f"  VSS  = EEV - RP  : {_fmt(self.vss)}" + _percent(self.relative_vss)
            )
        elif self.eev_infeasible:
            lines.append("  VSS  = EEV - RP  : unbounded (EEV infeasible)")
        return "\n".join(lines)

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "measure": ["WS", "RP", "EEV", "EVPI", "VSS"],
                "value": [
                    self.ws,
                    self.rp.objective,
                    self.eev,
                    self.evpi,
                    self.vss,
                ],
            }
        )

    def plot(self, path=None, *, theme: str = "light"):
        """Chart the three values and the two gaps. Requires matplotlib."""
        from .viz import plot_information_value

        if path is None and self.output_dir is not None:
            path = Path(self.output_dir) / "information_value.png"
        return plot_information_value(self, path, theme=theme)

    def __repr__(self) -> str:
        return f"SpAnalysis(name={self.name!r}, evpi={self.evpi!r}, vss={self.vss!r})"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.6g}"


def _percent(fraction: float | None) -> str:
    return "" if fraction is None else f"  ({fraction:.2%} of RP)"


def analyze(
    program,
    *,
    measures: Sequence[str] = MEASURES,
    mean_scenario_data: Dict[str, Any] | None = None,
) -> SpAnalysis:
    """Run the reference problems and derive the measures.

    Args:
        program: The StochasticProgram to analyse. Solved here if it has
            not been solved already.
        measures: Which of 'evpi' and 'vss' to compute. Each needs its own
            extra solves — EVPI one per scenario, VSS two — so asking for
            only one skips the other's work.
        mean_scenario_data: Entries to use for the expected-value
            problem's data instead of the averaged ones.

    Returns:
        An SpAnalysis. The constituent results stay on it, so the
        expected-value decision itself is reachable as `analysis.ev`.
    """
    unknown = set(measures) - set(MEASURES)
    if unknown:
        raise ValueError(
            f"measures must be drawn from {MEASURES}, got {sorted(unknown)}"
        )
    if not measures:
        raise ValueError("No measures requested")

    rp = program.solve()
    analysis = SpAnalysis(
        name=program.name,
        is_maximize=program.is_maximize,
        rp=rp,
        output_dir=program.output_dir,
    )

    if "evpi" in measures:
        analysis.scenario_values = _wait_and_see(program)
        analysis.ws = sum(
            scenario.probability * analysis.scenario_values[scenario.name]
            for scenario in program.scenarios
        )

    if "vss" in measures:
        analysis.ev = _expected_value(program, mean_scenario_data)
        analysis.eev, analysis.eev_infeasible = _expected_result_of_ev(
            program, analysis.ev
        )

    return analysis


def _sub_program(program, scenarios: Sequence[Scenario], suffix: str):
    """A deterministic-equivalent program over `scenarios`.

    Built from the same two builder functions as the program it comes
    from, so the reference problems are the same model — only the
    scenarios differ.
    """
    from .problem import StochasticProgram

    sub = StochasticProgram(
        f"{program.name}_{suffix}",
        sense="max" if program.is_maximize else "min",
        method="de",
        solver=program.solver_config.solver_name,
        solver_options=program.solver_config.kwargs,
        output_dir=Path(program.output_dir) / suffix,
        validate=program.validate,
        log_level=program.log_level,
    )
    sub.first_stage(program._first_stage_fn, state=program._state_names)
    sub.recourse(program._recourse_fn)
    sub.set_scenarios(ScenarioSet(list(scenarios), normalize=True))
    return sub


def _wait_and_see(program) -> Dict[str, float]:
    """Each scenario's optimum, solved as if that scenario were certain."""
    values: Dict[str, float] = {}
    for scenario in program.scenarios:
        alone = Scenario(scenario.name, 1.0, scenario.data)
        sub = _sub_program(program, [alone], f"ws_{scenario.name}")
        values[scenario.name] = sub.solve().objective
    return values


def _expected_value(program, mean_scenario_data) -> SpResult:
    """The mean-value problem: one scenario, the expectation of the data."""
    average = mean_scenario(program.scenarios, mean_scenario_data)
    return _sub_program(program, [average], "ev").solve()


def _expected_result_of_ev(program, ev: SpResult):
    """The expected-value decision, priced against every scenario.

    The first stage is fixed at what the mean-value problem chose and the
    deterministic equivalent is solved over the real scenario set, so the
    recourse absorbs whatever that decision failed to anticipate.

    Returns:
        (value, infeasible). A decision that cannot be carried out in
        some scenario yields (None, True) — VSS is then unbounded, which
        is a stronger statement than a large one.
    """
    sub = _sub_program(program, list(program.scenarios), "eev")
    built = sub.build()

    decision = [ev.first_stage_flat.get(label) for label in built.labels]
    if any(value is None for value in decision):
        raise ValueError(
            "The expected-value problem returned no value for part of the "
            "first-stage decision, so it cannot be evaluated against the "
            "scenarios."
        )
    for var, value in zip(built.root_solver.get_vars(), decision):
        var.fix(value)

    try:
        return sub.solve().objective, False
    except RuntimeError:
        # _solve_deterministic_equivalent raises when the solver does not
        # reach optimality, which with the first stage fixed means the
        # decision is not feasible for every scenario.
        return None, True


def measures_frame(analyses: List[SpAnalysis]):
    """One row per analysis, for comparing several programs."""
    import pandas as pd

    return pd.DataFrame(
        {
            "name": [a.name for a in analyses],
            "WS": [a.ws for a in analyses],
            "RP": [a.rp.objective for a in analyses],
            "EEV": [a.eev for a in analyses],
            "EVPI": [a.evpi for a in analyses],
            "VSS": [a.vss for a in analyses],
        }
    )

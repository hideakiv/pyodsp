"""Two-stage stochastic programming on top of pyodsp's decomposition algorithms.

Describe the problem once per stage and let the pipeline decompose it::

    import pyomo.environ as pyo
    from pyodsp.model.sp import StochasticProgram

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

    sp.set_scenarios(records)
    result = sp.solve()
    print(result.summary())

What the pipeline takes care of, none of which is visible above: choosing
between Benders decomposition, Benders with scaled cuts and dual
decomposition; replicating the first-stage variables into each scenario
and keeping the two coupling lists in the same order; converting a
maximize program to the minimize form the algorithms require and
converting every result back; and finding a valid bound for each
subproblem.
"""

from pyodsp.alg.risk import CVaR, Expectation

from .analysis import SpAnalysis, analyze
from .mean import mean_scenario
from .problem import StochasticProgram
from .result import ScenarioOutcome, SpResult
from ..scenario import Scenario, ScenarioSet
from ..state import StateSpec, StateView

__all__ = [
    "CVaR",
    "Expectation",
    "Scenario",
    "SpAnalysis",
    "analyze",
    "mean_scenario",
    "ScenarioOutcome",
    "ScenarioSet",
    "SpResult",
    "StateSpec",
    "StateView",
    "StochasticProgram",
]

"""Multistage stochastic programming on top of pyodsp's SDDP.

Describe one stage; the pipeline builds the lattice:

    import pyomo.environ as pyo
    from pyodsp.model.msp import MultistageProgram

    msp = MultistageProgram("inventory", sense="min", stage_bound=0.0)

    @msp.stage(state=["inventory"])
    def stage(m, state, node):
        m.inventory = pyo.Var(bounds=(0, 100))       # passed on
        m.buy = pyo.Var(bounds=(0, 50))
        m.balance = pyo.Constraint(
            expr=m.inventory == state.inventory + m.buy - node["demand"]
        )
        return COST * m.buy

    msp.set_initial_state(inventory=20.0)
    msp.set_realizations(realizations, stages=4)
    result = msp.solve()

`state.x` is what the previous stage left and `m.x` is what this one
leaves — the same quantity at two times, which is what consecutive
stages couple on. At stage 0 there is no previous stage, so `state.x` is
the initial condition.

For a two-stage problem stated as here-and-now plus recourse, use
pyodsp.model.sp.StochasticProgram instead.
"""

from .lattice import (
    LatticeNode,
    ScenarioLattice,
    independent,
    markov,
    stage_varying,
)
from .problem import MultistageProgram
from .result import MspResult

__all__ = [
    "LatticeNode",
    "MspResult",
    "MultistageProgram",
    "ScenarioLattice",
    "independent",
    "markov",
    "stage_varying",
]

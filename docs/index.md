# pyodsp

A Pyomo interface for decomposition of structured programs, inspired by
[DSP](https://github.com/Argonne-National-Laboratory/DSP).

pyodsp has two layers, and which one you want depends on how much of the
decomposition you want to own.

**The modelling front-end** ({doc}`guide/two-stage`, {doc}`guide/multistage`)
takes a stochastic program described once per stage and decomposes it for you:
it replicates the first-stage variables into every scenario, keeps the coupling
lists aligned, converts a maximize program into the minimize form the
algorithms require and converts every result back, finds a valid bound for each
subproblem, and picks the algorithm. This is the layer to start from.

**The decomposition algorithms** ({doc}`guide/low-level`) are Benders
decomposition, Benders with scaled cuts, dual decomposition and SDDP, driven
through a node graph you assemble yourself. Use this when the problem is not a
stochastic program, or when the structure you want to exploit is not the one
the front-end assumes.

## A two-stage program

```python
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

sp.set_scenarios({name: {"yield": y} for name, y in YIELDS.items()})
result = sp.solve()
print(result.summary())
```

`state.acres` is the scenario's own copy of the first-stage decision. Nothing
above says how the problem is split up, and that is the point: see
{doc}`guide/choosing-a-method` for what the pipeline decides on your behalf and
when you should overrule it.

## A multistage program

```python
from pyodsp.model.msp import MultistageProgram

msp = MultistageProgram("inventory", sense="min", stage_bound=0.0)

@msp.stage(state=["inventory"])
def stage(m, state, node):
    m.inventory = pyo.Var(bounds=(0, 100))     # what this stage passes on
    m.buy = pyo.Var(bounds=(0, 50))
    m.balance = pyo.Constraint(
        expr=m.inventory == state.inventory + m.buy - node["demand"]
    )
    return COST * m.buy

msp.set_initial_state(inventory=20.0)
msp.set_realizations(realizations, stages=4)
result = msp.solve()
```

One builder describes every stage. `state.inventory` is what the previous stage
left and `m.inventory` is what this one leaves — the same quantity at two
times, which is what consecutive stages couple on.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
guide/two-stage
guide/multistage
```

```{toctree}
:maxdepth: 2
:caption: Guide

guide/choosing-a-method
guide/scenarios
guide/risk
guide/analysis
guide/results
guide/mpi
guide/low-level
```

```{toctree}
:maxdepth: 2
:caption: Reference

examples
api/index
```

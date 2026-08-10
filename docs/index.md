# pyodsp

A Pyomo interface for decomposition of structured programs, inspired by
[DSP](https://github.com/Argonne-National-Laboratory/DSP).

**Decomposition is what pyodsp is.** {doc}`guide/low-level` describes the core:
Benders decomposition, Benders with scaled cuts, dual decomposition and SDDP,
driven through a node graph you assemble yourself — each node owning a Pyomo
model, an algorithm object and a list of coupling variables, with a runner
passing messages between them. Use this layer directly when the structure you
want to exploit is yours to describe.

**Every algorithm also runs under MPI** ({doc}`guide/mpi`), which spreads the
subproblems across ranks so a rank never builds a model it does not solve.

**Stochastic programming is an application of that layer.** The modelling
front-end ({doc}`guide/two-stage`, {doc}`guide/multistage`) takes a program
described once per stage and builds the decomposition for you: it replicates
the first-stage variables into every scenario, keeps the coupling lists
aligned, converts a maximize program into the minimize form the algorithms
require and converts every result back, finds a valid bound for each
subproblem, and picks the algorithm. Scenarios are where the block structure
comes from, so this is the layer to start from when your problem is a
stochastic program.

## A decomposition, wired by hand

```python
from pathlib import Path
import pyomo.environ as pyo

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.bd.alg_root_bm import BdAlgRootBm
from pyodsp.dec.bd.alg_leaf_pyomo import BdAlgLeafPyomo
from pyodsp.dec.bd.run import BdRun


def create_root_node():
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(within=pyo.NonNegativeReals)
    m.x2 = pyo.Var(within=pyo.NonNegativeReals)
    m.c1 = pyo.Constraint(expr=m.x1 + m.x2 <= 120)
    m.obj = pyo.Objective(expr=100 * m.x1 + 150 * m.x2, sense=pyo.minimize)

    coupling_dn = [m.x1, m.x2]                       # what the children see
    solver = PyomoSolver(m, SolverConfig(solver_name="appsi_highs"), coupling_dn)
    return DecNodeRoot(0, BdAlgRootBm(solver))


q1 = {1: -24, 2: -28}                                # per-block data


def create_leaf_node(i):
    b = pyo.ConcreteModel()
    b.x1 = pyo.Var(within=pyo.Reals)                 # this block's own copy
    b.x2 = pyo.Var(within=pyo.Reals)
    b.y1 = pyo.Var(within=pyo.NonNegativeReals)
    b.c1 = pyo.Constraint(expr=6 * b.y1 <= 60 * b.x1)
    b.obj = pyo.Objective(expr=q1[i] * b.y1, sense=pyo.minimize)

    coupling_up = [b.x1, b.x2]                       # matches coupling_dn, by position
    solver = PyomoSolver(b, SolverConfig(solver_name="appsi_highs"), coupling_up)
    node = DecNodeLeaf(i, BdAlgLeafPyomo(solver))
    node.set_bound(-30000.0)                         # valid bound on this block
    return node


root = create_root_node()
leaves = [create_leaf_node(1), create_leaf_node(2)]

root.add_child(1, multiplier=0.4)                    # weight in the master
root.add_child(2, multiplier=0.6)
root.set_groups([[1, 2]])                            # cut aggregation groups

BdRun([root, *leaves], Path("output/bd/optimality")).run()
```

The root holds the complicating variables and the leaves hold the blocks that
couple to them; `BdRun` passes trial points down and cuts back up until the
bound closes. `examples/bd/optimality.py` is this, complete and runnable.
Nothing here is stochastic — Benders only needs the block structure, and the
weights happen to be probabilities in this instance. {doc}`guide/low-level`
covers what you are responsible for once you wire it yourself: coupling order,
replica domains, sense and bounds.

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

guide/low-level
guide/mpi
guide/choosing-a-method
guide/scenarios
guide/risk
guide/analysis
guide/results
```

```{toctree}
:maxdepth: 2
:caption: Reference

examples
api/index
```

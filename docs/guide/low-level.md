# Driving the algorithms directly

{mod}`pyodsp.dec` is the core of pyodsp: the algorithms themselves, and the
node graph they run on. The stochastic programming front-end is one application
built on top of it. Drive it directly when the problem is not a two-stage or
multistage stochastic program, when the structure you want to exploit is not
the one the front-end assumes, or when you need to distribute a two-stage
problem across MPI ranks — which
{class}`~pyodsp.model.sp.problem.StochasticProgram` does not expose.

The price is that everything the front-end would have done for you becomes
yours to get right. This page is mostly about what those things are.

## The shape of a run

A decomposition is a graph of **nodes**. Each node owns a Pyomo model, an
algorithm object that knows how to talk to its neighbours, and a list of
coupling variables. A runner drives the message passing.

```python
from pathlib import Path
import pyomo.environ as pyo

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.bd.alg_root_bm import BdAlgRootBm
from pyodsp.dec.bd.alg_leaf_pyomo import BdAlgLeafPyomo
from pyodsp.dec.bd.run import BdRun


def create_root_node(solver="appsi_highs"):
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(within=pyo.NonNegativeReals)
    m.x2 = pyo.Var(within=pyo.NonNegativeReals)
    m.c1 = pyo.Constraint(expr=m.x1 + m.x2 <= 120)
    m.obj = pyo.Objective(expr=100 * m.x1 + 150 * m.x2, sense=pyo.minimize)

    coupling_dn = [m.x1, m.x2]                       # what the children see
    solver = PyomoSolver(m, SolverConfig(solver_name=solver), coupling_dn)
    return DecNodeRoot(0, BdAlgRootBm(solver))


def create_leaf_node(i, solver="appsi_highs"):
    block = pyo.ConcreteModel()
    block.x1 = pyo.Var(within=pyo.Reals)             # this scenario's own copy
    block.x2 = pyo.Var(within=pyo.Reals)
    block.y1 = pyo.Var(within=pyo.NonNegativeReals)
    block.c1 = pyo.Constraint(expr=6 * block.y1 <= 60 * block.x1)
    block.obj = pyo.Objective(expr=q1[i] * block.y1, sense=pyo.minimize)

    coupling_up = [block.x1, block.x2]               # must match coupling_dn, in order
    solver = PyomoSolver(block, SolverConfig(solver_name=solver), coupling_up)
    node = DecNodeLeaf(i, BdAlgLeafPyomo(solver))
    node.set_bound(-30000.0)
    return node


root = create_root_node()
leaves = [create_leaf_node(1), create_leaf_node(2)]

root.add_child(1, multiplier=0.4)                    # probability weights
root.add_child(2, multiplier=0.6)
root.set_groups([[1, 2]])                            # cut aggregation groups

BdRun([root, *leaves], Path("output/bd/optimality")).run()
```

`examples/bd/optimality.py` is this, complete and runnable.

## Seeing the output

pyodsp reports progress through the standard {mod}`logging` module, under the
`pyodsp` logger, and — like any well-behaved library — installs no handler of
its own. Until the application configures logging, a run is silent. The quickest
way in is the one helper pyodsp does expose:

```python
import pyodsp

pyodsp.configure_logging()              # INFO to stderr; pass logging.WARNING to quiet it
```

Anything the standard library can do works too — `logging.basicConfig()`, a
file handler on `logging.getLogger("pyodsp")`, or per-algorithm control through
children such as `pyodsp.dec.bd` and `pyodsp.alg.bm.<node id>`.

Each per-solver record carries its context and its numbers on `record.pyodsp` —
`{"node_id": ..., "depth": ...}`, plus an `"iteration"` dict of the row's fields
(`lb`, `ub`, `num_cuts`, `elapsed`, …) on the iteration lines. `configure_logging`
uses it to draw the `Node: <id> -` tree prefix; a custom handler can read the
numbers straight off the record instead of parsing the text.

## The four things you now own

### 1. Coupling lists match by position

This is the one to be careful about. A parent's `coupling_dn` and every child's
`coupling_up` are matched **by position, with no name check**. Two lists built
in different orders produce a decomposition that runs happily to convergence
and returns the wrong answer.

The front-end has a whole module — {mod}`pyodsp.model.state` — whose job is to
own one canonical flattening order so that every list is generated from it
rather than assembled by hand. Doing this yourself means generating both lists
from a single source, not writing them out twice.

### 2. Replicating the first stage

Each leaf declares its own copy of every coupling variable. Under Benders those
replicas should be **free** — `within=pyo.Reals`, no bounds — because they are
placeholders that the algorithm fixes to whatever trial point the master
produced, and a replica that kept an integer domain would reject the fractional
trial points an LP master legitimately produces.

Dual decomposition is the opposite case: its non-anticipativity constraints
have to tie together variables that can actually take the same values, so there
the replicas keep the real first-stage domain. `examples/sslp/dd.py` has
replicas that stay `Binary`.

### 3. Sense

The algorithms only accept minimization. Every model you build must be written
as one, and every number that comes back is in that form. The front-end negates
a maximize objective on the way in and converts every result back on the way
out; here there is nothing doing that for you.

### 4. Bounds

`DecNodeLeaf.set_bound(...)` gives the master a valid bound on that
subproblem's objective. Without it the first master iteration can be unbounded.
The front-end estimates one per scenario by solving the recourse objective over
the whole problem; here it is a number you supply, and it must be valid at
every trial point the master can produce, not just at the optimum.

## Cut aggregation

`root.set_groups([[1, 2]])` puts both leaves in one group, so their cuts are
aggregated into a single cut per iteration. `[[1], [2]]` keeps them separate —
more cuts per iteration, a tighter master, more work per solve. This is a real
tuning knob on problems with many scenarios.

## The pieces

```{list-table}
:header-rows: 1
:widths: 34 66

* - Module
  - What it holds
* - {mod}`pyodsp.dec.node.dec_node`
  - `DecNodeRoot`, `DecNodeLeaf`, `DecNodeInner` — the graph. (The first two
    are aliases for `DecNodeParent` and `DecNodeChild`.)
* - `pyodsp.dec.bd`
  - Benders: `BdAlgRootBm` (bundle-method master), `BdAlgLeafPyomo`, `BdRun`,
    `BdRunMpi`.
* - `pyodsp.dec.bdsc`
  - Benders with scaled cuts, for integer recourse: `BdScRun`, `BdScRunMpi`.
    After {ref}`van der Laan and Romeijnders (2024) <bdsc-citation>`.
* - `pyodsp.dec.dd`
  - Dual decomposition: `DdAlgRootBm`, `MipHeuristicRoot`, `DdRun`, `DdRunMpi`.
    Pass `mode="proximal"` to the root algorithm for the proximal bundle
    method, which needs a quadratic-capable solver — see
    `examples/dd/equality_pbm.py`.
* - `pyodsp.dec.sddp`
  - SDDP: `SddpRun`, `SddpRunMpi`, `SddpPolicy`.
* - `pyodsp.dec.graph`
  - The message-passing topologies: `HubAndSpoke` for two-stage, `Lattice` for
    multistage, `Tree`, and MPI variants of the first two.
* - {mod}`pyodsp.solver.pyomo_solver`
  - `PyomoSolver`, `SolverConfig` — the Pyomo wrapper every node holds, and
    where the maximize-to-minimize conversion lives.
```

## Worked examples

`examples/bd/`, `examples/dd/` and `examples/bdsc/` drive the algorithms on
small models, in serial and MPI form. `examples/farmer/bd_model.py` is the
farmer wired by hand — the direct comparison against
`examples/farmer/sp_pipeline.py`, which is the same problem through the
front-end. `examples/balance/sddp.py` and `examples/aircon/sddp.py` are
hand-wired multistage runs, against `examples/inventory/msp_pipeline.py`
through the front-end.

Reading a pair side by side is the fastest way to see what the front-end
actually does.

# Multistage programs

More than two stages, decisions and observations alternating over a horizon.
{class}`~pyodsp.model.msp.problem.MultistageProgram` solves these by SDDP, and asks you
to describe **one stage**: the same builder runs at every node of the lattice,
told which stage and which realization it is building.

```python
import pyomo.environ as pyo
from pyodsp.model.msp import MultistageProgram

msp = MultistageProgram("inventory", sense="min", stage_bound=0.0)

@msp.stage(state=["inventory"])
def stage(m, state, node):
    m.inventory = pyo.Var(bounds=(0, CAPACITY))    # what this stage passes on
    m.bought = pyo.Var(bounds=(0, CAPACITY))
    m.shortage = pyo.Var(bounds=(0, CAPACITY))

    demand = node.get("demand", 0.0)
    m.balance = pyo.Constraint(
        expr=m.inventory == state.inventory + m.bought - demand + m.shortage
    )

    price = BUY_NOW if node.is_first else BUY_LATER
    return price * m.bought + PENALTY * m.shortage

msp.set_initial_state(inventory=0.0)
msp.set_realizations(REALIZATIONS, stages=4)
result = msp.solve()
```

## The state appears twice

This is the one idea the whole front-end turns on. A state variable is the same
quantity at two different times, and both are visible inside the builder:

`state.inventory`
: What the **previous** stage left. At stage 0 there is no previous stage, so
  this is the initial condition — a plain number rather than a variable, which
  is exactly why one builder can cover every stage without branching.

`m.inventory`
: What **this** stage leaves. A variable you declare, and the thing the next
  stage will receive.

`state=[...]` is required, unlike in a two-stage program. A stage model holds
its own decisions as well as the state, so there is nothing sensible to infer;
the builder has to say which of its variables carry over. `state=[]` is
rejected too — it leaves the stages uncoupled, which is a series of unrelated
problems rather than a multistage one.

## `node`

The third argument is a {class}`~pyodsp.model.msp.lattice.LatticeNode`:

```{list-table}
:header-rows: 1
:widths: 22 78

* - Attribute
  - Meaning
* - `node.stage`
  - Which stage, counting from 0.
* - `node.index`
  - Position within that stage.
* - `node.name`
  - Its name, unique within the stage.
* - `node.idx`
  - `"{stage}-{index}"` — the id the decomposition layer uses, and the one
    `result.simulate` takes.
* - `node.is_first`
  - True at stage 0, whose state is the initial condition.
* - `node.is_last`
  - True at the final stage, which passes no state on.
* - `node["demand"]`
  - The realized data. `node.get(key, default)` and attribute access work too.
```

Stage 0 has no realization of its own — the first decision is made before
anything is observed — so `node["demand"]` there raises unless you supplied
`first_stage_data`. Use `node.get("demand", 0.0)` or branch on `node.is_first`.
Deterministic data that merely *varies* by stage needs nothing special: index
it with `node.stage`.

## `stage_bound`

SDDP prices each stage's cost-to-go with a variable that starts out unbounded.
Every stage needs a bound on it before its parent can price it; without one the
first master is unbounded. `stage_bound` is that number, in your own units — a
lower bound when minimizing.

For a minimization where no stage can cost less than nothing, `stage_bound=0.0`
is both valid and tight enough to work with. There is no automatic estimate
here, unlike the two-stage `auto_bound`.

## The scenario lattice

SDDP here works on a **recombining lattice**, not a scenario tree: each stage
has its own set of realizations, and every node at stage $t$ leads to every
node at stage $t+1$ with a transition probability. That is what keeps the
problem size linear in the horizon — a tree would multiply out.

Four ways to state it, in increasing generality:

### Stage-wise independent

```python
msp.set_realizations(
    [{"name": "low",  "probability": 0.3, "demand": 10.0},
     {"name": "mid",  "probability": 0.5, "demand": 25.0},
     {"name": "high", "probability": 0.2, "demand": 40.0}],
    stages=4,
)
```

The same realizations at every stage after the first, with the same
probabilities regardless of what came before. `stages` counts the first stage,
so four means one here-and-now decision followed by three observed stages. This
is what most textbook multistage models assume.

### Stage-varying

```python
msp.set_stage_realizations([spring, summer, autumn])   # one per stage after the first
```

The distribution — and the number of realizations — may differ at every stage.
Inflows wetter in spring than in summer, demand that fans out as the horizon
recedes. Left without `transitions`, each stage is drawn independently of the
last with its own probabilities.

### Markov

```python
msp.set_markov_realizations(
    [{"name": "dry", "demand": 10.0}, {"name": "wet", "demand": 40.0}],
    stages=6,
    transition_matrix=[[0.8, 0.2], [0.3, 0.7]],
)
```

The realizations are states of a chain, so where you go next depends on where
you are — a dry month tending to follow a dry month. `initial_distribution`
says how stage 1 is reached from the first stage; the realization probabilities
are used when it is omitted.

### Directly

```python
from pyodsp.model.msp import ScenarioLattice
msp.set_lattice(ScenarioLattice(stages, transitions))
```

`transitions[t][i][j]` is the probability of moving from node `i` of stage `t`
to node `j` of stage `t+1`. One matrix per stage except the last, each row
summing to one. Stage 0 must hold exactly one node.

Whichever you use, look at it before committing to a run — the lattice is where
a multistage model's size comes from:

```python
msp.plot_scenario_lattice("lattice.png")   # needs matplotlib
print(msp.describe())
```

## The answer is a policy

SDDP does not return a schedule. What it leaves behind is a set of cuts per
stage describing what the future costs, and any scenario path can be walked
against them afterwards:

```python
result = msp.solve()
print(result.summary())

for step in result.simulate(["0-0", "1-2", "2-2", "3-2"]):
    print(step.node_idx, step.stage_cost, step.next_state)
```

```text
inventory: minimize via sddp
  bound      : 174.75
  stages     : 4 (nodes per stage [1, 3, 3, 3])
  iterations : 20
  first stage:
    inventory = 75

0-0 150.0 [75.0]
1-2 0.0 [35.0]
2-2 75.0 [10.0]
3-2 150.0 []
```

That path is the high-demand branch at every stage: the policy stocks 75 up
front at the cheap price, coasts through the first high month without buying,
and only then starts paying the late price. The final stage passes no state on,
so `next_state` is empty there.

The path is a list of node ids, `"{stage}-{index}"`. `result.policy()` returns
the {class}`~pyodsp.dec.sddp.policy.SddpPolicy` directly if you want to keep it
around.

## Convergence

SDDP approaches the optimum from one side and estimates the other by
simulation, so `result.bound` is a bound and not a proven optimum. Every
`sample_frequency` iterations the run stops and evaluates `sample_size`
scenario paths, producing a confidence interval at `confidence_level` to
compare the bound against. `result.simulation` has one row per such test, and
`result.objective_stats()` summarizes the last sample.

```python
msp = MultistageProgram(..., sample_frequency=10, sample_size=100, confidence_level=0.95)
```

A wide interval means the sample, not the policy, is the uncertain part —
raise `sample_size`.

## Running under MPI

```python
msp = MultistageProgram(..., mpi=True)
```

See {doc}`mpi`. In short: every rank builds the identical lattice and helps
evaluate the Monte Carlo sample, rank 0 drives the iteration, and only rank 0's
result carries the answer — check `result.is_root_rank` before reporting.

## Full API

Every argument and method of {class}`~pyodsp.model.msp.problem.MultistageProgram` is
documented in {doc}`../api/model`.

## References

(sddp-citation)=

For the method and the variants it has grown since Pereira and Pinto
introduced it:

> Füllner, C., & Rebennack, S. (2025). Stochastic dual dynamic programming and
> its variants: A review. *SIAM Review*, 67(3), 415–539.

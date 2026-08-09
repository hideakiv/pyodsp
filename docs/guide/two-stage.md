# Two-stage programs

A two-stage stochastic program has one here-and-now decision, then the
uncertainty resolves, then a recourse decision that may depend on what was
observed. {class}`~pyodsp.model.sp.problem.StochasticProgram` asks you to describe each
of those once — the first stage, and *one representative scenario's* recourse —
and builds the rest.

```python
import pyomo.environ as pyo
from pyodsp.model.sp import StochasticProgram

sp = StochasticProgram("farmer", sense="max")
```

`sense` is stated in your own terms. The algorithms only accept minimization;
a maximize program is negated on the way in and every number you get back is
converted to your sense again, so nothing in your model or your results ever
needs a sign flip.

## The first stage

```python
@sp.first_stage
def first_stage(m):
    m.acreage = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
    m.land = pyo.Constraint(expr=sum(m.acreage[c] for c in CROPS) <= 500)
    return -sum(PLANTING_COST[c] * m.acreage[c] for c in CROPS)
```

The builder receives an empty {class}`~pyomo.core.base.PyomoModel.ConcreteModel`,
declares variables and constraints on it, and **returns the first-stage
objective expression**. Do not declare a `pyo.Objective` yourself — the
pipeline installs one, in the right sense, and rejects a model that already
carries one.

## The recourse

```python
@sp.recourse
def recourse(m, state, scenario):
    m.sold = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
    m.purchased = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
    m.balance = pyo.Constraint(
        CROPS,
        rule=lambda m, c: m.sold[c] - m.purchased[c]
        == scenario["yield"][c] * state.acreage[c] - FEED[c],
    )
    return sum(PRICE[c] * m.sold[c] - COST[c] * m.purchased[c] for c in CROPS)
```

This runs **once per scenario**, and gets three arguments:

`m`
: An empty model, this scenario's own.

`state`
: A {class}`~pyodsp.model.state.StateView`. `state.acreage` is *this scenario
  model's own copy* of the first-stage variable, not the first-stage variable
  itself. The pipeline creates the replica and couples it; you just write
  constraints against it. It is read-only — declaring a recourse variable means
  putting it on `m`.

`scenario`
: A {class}`~pyodsp.model.scenario.Scenario`. Its data is reachable three
  ways — `scenario["yield"]`, `scenario.get("yield", default)`, and attribute
  access for any key that is a valid identifier (`scenario.demand`) — alongside
  `scenario.name` and `scenario.probability`, which keep priority over data
  entries of the same name.

The return value is the recourse objective **for this scenario alone**,
unweighted. Probability weighting is the pipeline's job.

### What the state is, by default

Every variable declared on the first-stage model. That is what a two-stage
program normally wants: the recourse sees the whole here-and-now decision.

To narrow it:

```python
@sp.first_stage(state=["acreage"])
def first_stage(m):
    m.acreage = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
    m.budget_slack = pyo.Var(domain=pyo.NonNegativeReals)   # not passed down
    ...
```

A narrowed state is only safe under Benders, whose master keeps the whole first
stage to itself. BDSC and dual decomposition embed the first stage inside every
scenario, and a first-stage variable left out of the state vector then gets an
uncoupled copy per scenario — a here-and-now decision quietly turned into a
wait-and-see one. Both algorithms refuse to build in that case rather than
returning a wrong number. See {doc}`choosing-a-method`.

## Scenarios

```python
sp.set_scenarios({"good": {"yield": ...}, "poor": {"yield": ...}})
```

`set_scenarios` accepts a {class}`~pyodsp.model.scenario.ScenarioSet`, a pandas
DataFrame with one row per scenario, a mapping of name to data, or an iterable
of `Scenario` or of dicts. Probabilities default to uniform. The alternative is
to append them one at a time:

```python
sp.add_scenario("good", yield_=..., probability=0.3)
sp.add_scenario("poor", yield_=..., probability=0.7)
```

Either every scenario carries a probability or none does — a partial set is
rejected rather than half-filled. Probabilities must sum to one unless you pass
`normalize=True`. {doc}`scenarios` covers the input formats in full.

## Solving

```python
result = sp.solve()
print(result.summary())
```

```text
farmer: maximize via bd
  objective : 108390
  bound     : 108390
  gap       : 0.0000%
  iterations: 7
  first stage:
    acreage[WHEAT] = 170
    acreage[CORN] = 80
    acreage[BEETS] = 250
  scenarios:
    GOOD (p=0.3333): 275900
    AVERAGE (p=0.3333): 218250
    POOR (p=0.3333): 157720
```

{class}`~pyodsp.model.sp.result.SpResult` carries the first-stage decision, the
per-scenario outcomes, the bound and gap, and the per-iteration history — all
in your own units. {doc}`results` goes through it.

### Looking before you leap

`build()` constructs the whole node graph without running anything, and
`describe()` reports what the pipeline decided:

```python
print(sp.describe())
```

```text
Stochastic program 'farmer'
  sense           : maximize
  algorithm       : bd
  scenarios       : 3
  state variables : 3 (acreage[WHEAT], acreage[CORN], acreage[BEETS])
  risk            : expectation
  solver          : appsi_highs
  output          : output/farmer
```

Worth doing on any model you have not run before: if the algorithm line does
not say what you expected, {doc}`choosing-a-method` explains why.

## Bounds on the recourse

Benders needs a lower bound on each scenario's recourse cost before its master
can price it; without one the first master iteration is unbounded. By default
the pipeline computes one per scenario, by optimizing that scenario's recourse
objective over the *whole* problem, first stage included — so the answer is
valid at every trial point the master can produce.

That costs one extra solve per scenario at build time. If you know a bound:

```python
sp = StochasticProgram(..., recourse_bound=0.0)   # in your own units
```

or turn the estimate off with `auto_bound=False` and accept an unbounded theta,
which is legal but can leave the first iteration unbounded. When the bounding
solve fails or comes back non-optimal you get a `RuntimeWarning` and the run
continues without a bound for that scenario.

## Validation

`validate=True` (the default) runs three structural checks:

- no recourse builder overwrote a state variable with a component of its own,
- no integer recourse variable reaches plain Benders,
- BDSC and dual decomposition got the complete state vector.

Each costs a pass over a built model, and the last two need a probe model of
their own. Turning them off on a large model you have already run once is
reasonable; nothing about the requirements changes, they simply stop being
enforced, and a violation then produces a confidently wrong answer instead of
an error.

## Output

Each node writes to `output/<name>/` unless `output_dir` says otherwise. The
per-iteration bound and incumbent trajectory that `result.history` exposes is
read back from there, so a finished run is inspectable after the fact.

## Full API

Every argument and method of {class}`~pyodsp.model.sp.problem.StochasticProgram` is
documented in {doc}`../api/model`.

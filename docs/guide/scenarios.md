# Scenarios

pyodsp models uncertainty as a finite list of realizations with probabilities.
Sampling and scenario reduction are out of scope: whatever produces your
scenarios, it hands the result here.

## The four input formats

`set_scenarios` coerces all of these through
{func}`~pyodsp.model.scenario.as_scenario_set`, so pick whichever matches where
your data already lives.

**A mapping of name to data.** Shortest when the probabilities are uniform:

```python
sp.set_scenarios({
    "GOOD":    {"yield": {"WHEAT": 3.0, "CORN": 3.6}},
    "AVERAGE": {"yield": {"WHEAT": 2.5, "CORN": 3.0}},
    "POOR":    {"yield": {"WHEAT": 2.0, "CORN": 2.4}},
})
```

**An iterable of dicts.** `name` and `probability` are read out as such; every
other key becomes scenario data:

```python
sp.set_scenarios([
    {"name": "low",  "probability": 0.3, "demand": 10.0},
    {"name": "high", "probability": 0.7, "demand": 40.0},
])
```

A record with no `name` is named by its position.

**A pandas DataFrame**, one row per scenario, same convention:

```python
frame = pd.read_csv("scenarios.csv")   # columns: name, probability, demand, price
sp.set_scenarios(frame)
```

**A ScenarioSet**, built explicitly, when you want the validation to happen at
a point you choose:

```python
from pyodsp.model.sp import Scenario, ScenarioSet

scenarios = ScenarioSet([
    Scenario("low", 0.3, {"demand": 10.0}),
    Scenario("high", 0.7, {"demand": 40.0}),
])
sp.set_scenarios(scenarios)
```

Or one at a time, which merges a `data` mapping with any keyword arguments:

```python
sp.add_scenario("low", demand=10.0, probability=0.3)
sp.add_scenario("high", {"demand": 40.0}, probability=0.7)
```

## The rules

**Probabilities sum to one**, or the set is rejected. `normalize=True` rescales
instead:

```python
sp.set_scenarios(records, normalize=True)
```

**All or nothing.** Either every scenario carries a probability or none does; a
partial set has no sensible completion, so it raises rather than guessing. With
none given, the scenarios end up equally likely.

**Names are unique**, and negative probabilities are rejected.

**Order is meaningful.** It fixes the leaf node indices, so a rerun of the same
ScenarioSet writes its per-scenario output to the same places.

## Reading data in the builder

The recourse builder gets the {class}`~pyodsp.model.scenario.Scenario` itself,
not just its data:

```python
@sp.recourse
def recourse(m, state, scenario):
    demand = scenario["demand"]              # subscript
    price = scenario.get("price", 1.0)       # with a default
    label = scenario.name                    # the name and probability
    ...
```

Attribute access falls through to the data for any key that is a valid Python
identifier, but `name`, `probability` and `data` keep their own meaning — a
data entry called `name` is reachable only as `scenario["name"]`.

There is no schema. Whatever you put in the data comes back untouched, which is
why nested structures like `{"yield": {"WHEAT": 3.0}}` work without ceremony.

## Multistage realizations

{class}`~pyodsp.model.msp.problem.MultistageProgram` uses the same formats, one
collection per stage rather than one for the problem, and wraps them in a
lattice. See {doc}`multistage`; the difference that matters is that a
multistage node also knows its stage and position, and stage 0 has no
realization at all.

## Reference

{class}`~pyodsp.model.scenario.Scenario`,
{class}`~pyodsp.model.scenario.ScenarioSet` and its constructors are documented
in {doc}`../api/model`.

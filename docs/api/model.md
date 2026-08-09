# `pyodsp.model` — the modelling front-end

```{eval-rst}
.. automodule:: pyodsp.model
   :no-members:
```

## Two-stage

```{eval-rst}
.. automodule:: pyodsp.model.sp
   :no-members:
```

### Problem

```{eval-rst}
.. automodule:: pyodsp.model.sp.problem
   :members:
   :member-order: bysource
```

### Results

```{eval-rst}
.. automodule:: pyodsp.model.sp.result
   :members: SpResult, ScenarioOutcome, read_result
   :member-order: bysource
```

### Analysis

```{eval-rst}
.. automodule:: pyodsp.model.sp.analysis
   :members:
   :member-order: bysource
```

### The mean scenario

```{eval-rst}
.. automodule:: pyodsp.model.sp.mean
   :members:
   :member-order: bysource
```

### Builders

The node graph each method produces. You do not call these directly —
`StochasticProgram.build` dispatches to one of them — but they are where the
requirements of each algorithm are enforced, so the docstrings explain the
rules {doc}`../guide/choosing-a-method` summarizes.

```{eval-rst}
.. automodule:: pyodsp.model.sp.builders
   :members:
   :member-order: bysource
```

### Plotting

```{eval-rst}
.. automodule:: pyodsp.model.sp.viz
   :members:
   :member-order: bysource
```

## Multistage

```{eval-rst}
.. automodule:: pyodsp.model.msp
   :no-members:
```

### Problem

```{eval-rst}
.. automodule:: pyodsp.model.msp.problem
   :members:
   :member-order: bysource
```

### The scenario lattice

```{eval-rst}
.. automodule:: pyodsp.model.msp.lattice
   :members:
   :member-order: bysource
```

### Results

```{eval-rst}
.. automodule:: pyodsp.model.msp.result
   :members: MspResult, read_result
   :member-order: bysource
```

### Builders

```{eval-rst}
.. automodule:: pyodsp.model.msp.builders
   :members:
   :member-order: bysource
```

### Plotting

```{eval-rst}
.. automodule:: pyodsp.model.msp.viz
   :members:
   :member-order: bysource
```

## Shared

### Scenarios

```{eval-rst}
.. automodule:: pyodsp.model.scenario
   :members:
   :member-order: bysource
```

### State variables

The state vector is the only thing the stages share, and getting its order
wrong is the classic way to build a decomposition that runs happily and returns
the wrong answer — the decomposition layer matches coupling lists by position,
with no name check. This module owns the canonical flattening order.

```{eval-rst}
.. automodule:: pyodsp.model.state
   :members:
   :member-order: bysource
```

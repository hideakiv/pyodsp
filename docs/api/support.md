# `pyodsp.alg`, `pyodsp.solver`, `pyodsp.viz`

```{eval-rst}
.. automodule:: pyodsp.alg
   :no-members:

.. automodule:: pyodsp.solver
   :no-members:

.. automodule:: pyodsp.viz
   :no-members:
```

## Risk measures

```{eval-rst}
.. automodule:: pyodsp.alg.risk
   :members:
   :member-order: bysource
```

## Bundle methods

The master algorithms every decomposition here is built on.

`BundleMethod`
: The plain cutting-plane-with-a-bundle. Benders, BDSC's root master, and
  dual decomposition in its default mode all run on this.

`ProximalBundleMethod`
: Adds a quadratic proximal term, so it needs a quadratic-capable solver.
  Used by dual decomposition under `mode="proximal"`, and by BDSC's
  cut-generation master, which is why `method='bdsc'` pulls in a second
  solver.

`CuttingPlaneMethod`
: The bundle-free cutting plane the two above are built over.

`RestrictedBundleMethod`
: Implemented, but not currently reached by any of the algorithms.

```{eval-rst}
.. automodule:: pyodsp.alg.bm.bm
   :members:
   :member-order: bysource

.. automodule:: pyodsp.alg.bm.pbm
   :members:
   :member-order: bysource

.. automodule:: pyodsp.alg.bm.rbm
   :members:
   :member-order: bysource

.. automodule:: pyodsp.alg.bm.cp
   :members:
   :member-order: bysource

.. automodule:: pyodsp.alg.bm.cuts
   :members:
   :member-order: bysource

.. automodule:: pyodsp.alg.bm.cuts_manager
   :members:
   :member-order: bysource
```

## Algorithm parameters

Tolerances and step-size constants, in one place.

```{eval-rst}
.. automodule:: pyodsp.alg.params
   :members:
   :member-order: bysource

.. automodule:: pyodsp.alg.const
   :members:
   :member-order: bysource
```

## The Pyomo wrapper

Every node holds a `PyomoSolver`. It is also where the maximize-to-minimize
conversion happens — the model is converted on construction and each solver
remembers that it flipped, which is what lets results come back in your own
units without anything tracking a sign.

```{eval-rst}
.. automodule:: pyodsp.solver.solver
   :members:
   :member-order: bysource

.. automodule:: pyodsp.solver.pyomo_solver
   :members:
   :member-order: bysource

.. automodule:: pyodsp.solver.sense
   :members:
   :member-order: bysource

.. automodule:: pyodsp.solver.pyomo_utils
   :members:
   :member-order: bysource
```

## Plotting

Requires matplotlib (`pip install -e ".[viz]"`), imported lazily.

```{eval-rst}
.. automodule:: pyodsp.viz.convergence
   :members:
   :member-order: bysource

.. automodule:: pyodsp.viz.style
   :members:
   :member-order: bysource
```

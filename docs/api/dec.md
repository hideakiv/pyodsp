# `pyodsp.dec` — decomposition

The layer underneath the modelling front-end: a graph of nodes, each owning a
model and an algorithm, driven by a runner. {doc}`../guide/low-level` is the
narrative version.

```{eval-rst}
.. automodule:: pyodsp.dec
   :no-members:
```

## Nodes

```{eval-rst}
.. automodule:: pyodsp.dec.node.dec_node
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.node.cut_aggregator
   :members:
   :member-order: bysource
```

### Interfaces

What a node and an algorithm have to implement, and the messages they exchange.
Private by name, but they are the vocabulary every signature below is written
in.

```{eval-rst}
.. automodule:: pyodsp.dec.node._node
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.node._alg
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.node._message
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.node._logger
   :members:
   :member-order: bysource
```

## Benders decomposition

```{eval-rst}
.. automodule:: pyodsp.dec.bd.message
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.bd.alg_root_bm
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.bd.alg_leaf_pyomo
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.bd.run
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.bd.run_mpi
   :members:
   :member-order: bysource
```

## Benders with scaled cuts

Handles integer recourse exactly. Rather than reading a cut off the LP duals —
which an integer second stage does not have — it builds cuts that recover the
convex hull of the recourse objective by row generation. The cut-generation
master runs a proximal bundle method, so it needs a quadratic-capable solver of
its own.

> van der Laan, N., & Romeijnders, W. (2024). A converging Benders'
> decomposition algorithm for two-stage mixed-integer recourse models.
> *Operations Research*, 72(5), 2190–2214.

```{eval-rst}
.. automodule:: pyodsp.dec.bdsc
   :no-members:

.. automodule:: pyodsp.dec.bdsc.message
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.bdsc.alg_root_bm
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.bdsc.alg_leaf_pyomo
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.bdsc.master_creator
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.bdsc.run
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.bdsc.run_mpi
   :members:
   :member-order: bysource
```

## Dual decomposition

```{eval-rst}
.. automodule:: pyodsp.dec.dd.message
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.dd.alg_root_bm
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.dd.alg_leaf_pyomo
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.dd.coupling_manager
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.dd.master_creator
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.dd.mip_heuristic_root
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.dd.run
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.dd.run_mpi
   :members:
   :member-order: bysource
```

## SDDP

> Füllner, C., & Rebennack, S. (2025). Stochastic dual dynamic programming and
> its variants: A review. *SIAM Review*, 67(3), 415–539.

```{eval-rst}
.. automodule:: pyodsp.dec.sddp
   :no-members:

.. automodule:: pyodsp.dec.sddp.policy
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.sddp.run
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.sddp.run_mpi
   :members:
   :member-order: bysource
```

## Graph topologies

How messages move: hub-and-spoke for two-stage problems, a lattice for
multistage ones.

```{eval-rst}
.. automodule:: pyodsp.dec.graph.hub_and_spoke
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.graph.hub_and_spoke_mpi
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.graph.lattice
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.graph.lattice_mpi
   :members:
   :member-order: bysource

.. automodule:: pyodsp.dec.graph.tree
   :members:
   :member-order: bysource
```

## Utilities

```{eval-rst}
.. automodule:: pyodsp.dec.utils
   :members:
   :member-order: bysource
```

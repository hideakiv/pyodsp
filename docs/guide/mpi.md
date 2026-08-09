# Running under MPI

Every algorithm has an MPI runner alongside its serial one. They come in two
kinds, and the difference matters for how you write the script.

Install [mpi4py](https://mpi4py.readthedocs.io/en/stable/) and
[MPICH](https://www.mpich.org/) first; OpenMPI is not tested. mpi4py is
imported only by the `run_mpi` modules, so a serial installation never
touches it.

## Two-stage: nodes spread across ranks

{class}`~pyodsp.dec.bd.run_mpi.BdRunMpi`,
{class}`~pyodsp.dec.bdsc.run_mpi.BdScRunMpi` and
{class}`~pyodsp.dec.dd.run_mpi.DdRunMpi` split **distinct nodes** across ranks.
Each rank builds only the nodes it owns and passes them to the runner as a
list — usually a list of one:

```python
from mpi4py import MPI
from pyodsp.dec.bd.run_mpi import BdRunMpi

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    node = create_root_node()
    node.add_child(1, multiplier=0.4)
    node.add_child(2, multiplier=0.6)
if rank == 1:
    node = create_leaf_node(1)
if rank == 2:
    node = create_leaf_node(2)

BdRunMpi([node], Path("output/bd/optimality_mpi")).run()

if rank == 0:
    ...   # rank 0 has the master, and the answer

MPI.Finalize()
```

```bash
mpiexec -n 3 python optimality_mpi.py
```

Rank 0 holds the root; the leaves are distributed over the rest. The point is
that a rank never builds a model it does not solve, so the memory a scenario
costs is paid once, on one rank — which is what makes this worth doing on a
scenario set too large for one machine.

{class}`~pyodsp.model.sp.problem.StochasticProgram` does not expose this. The two-stage
front-end builds every node in one process, so to distribute a two-stage
problem you drop to {doc}`low-level`.

## Multistage: every rank builds the lattice

SDDP is different, and the front-end does expose it:

```python
msp = MultistageProgram("inventory", ..., mpi=True)
result = msp.solve()

if result.is_root_rank:
    print(result.summary())
```

```bash
mpiexec -n 4 python examples/inventory/msp_pipeline_mpi.py
```

Here **every rank builds the identical lattice**. Rank 0 drives the iteration;
the other ranks help evaluate the Monte Carlo sample that tests convergence.
That is the parallelism on offer — the sample, not the stages — so the speedup
tracks `sample_size`, not the horizon.

Three consequences:

- **Every rank returns a result, and only rank 0's carries the answer.** The
  others come back with an empty bound and history. Guard on
  `result.is_root_rank` before printing or asserting anything.
- **Only rank 0 writes output.**
- **Only rank 0 purges cuts.** A replica that aged cuts out on its own schedule
  would stop matching the trial points rank 0 is solving, so the front-end
  makes cut purging conditional on `is_root_rank`.

`msp.rank` and `msp.is_root_rank` are available before the solve, and both are
well-defined without MPI — rank 0, always root.

## Choosing the rank count

How many ranks a two-stage run needs is up to how you deal the nodes out, and
there are two patterns in the examples.

**One node per rank.** `examples/bd/optimality_mpi.py` and
`examples/sslp/dd_mpi.py` assign rank 0 the master and one scenario to each
remaining rank, asserting `size == num_scenarios + 1` up front. Simple, and the
assertion earns its place: a mismatch otherwise shows up as a hang rather than
an error.

**Round-robin.** `examples/bdsc/cs_mpi.py` deals the scenarios over whatever
ranks it is given, which is the pattern to copy when the rank count is not
yours to choose:

```python
nodes = []
if rank == 0:
    root = create_root_node(R, solver)
    for s in range(1, R + 1):
        root.add_child(s, multiplier=1 / R)
    root.set_groups([[s for s in range(1, R + 1)]])
    nodes.append(root)

for s in range(1, R + 1):
    if (s - 1) % (size - 1) + 1 == rank:
        nodes.append(create_leaf_node(s, R, solver))

BdScRunMpi(nodes, Path("output/bdsc/cs_mpi")).run()
```

It runs on 2 ranks with one taking every scenario, on `R + 1` with one each, or
on more than that, where the surplus ranks pass an empty `nodes` list and simply
have nothing to do. All three are covered by the test suite. Note that the root
declares its children and their cut group even though the leaves live on other
ranks.

For SDDP any count works; the ranks are sample evaluators.

## Testing

`test/mpi/` holds the MPI tests. They need `mpiexec` on the path, and are
skipped otherwise.

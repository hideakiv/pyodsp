# Installation

```bash
git clone https://github.com/hideakiv/pyodsp
cd pyodsp
pip install -e .
```

That pulls in Pyomo, numpy, pandas, scipy and `highspy`, which is enough to run
every algorithm in serial.

## Extras

```bash
pip install -e ".[viz]"    # matplotlib, for the plot_* helpers
pip install -e ".[docs]"   # Sphinx, to build this documentation
```

Nothing in the solve path imports matplotlib. It is needed only by
{mod}`pyodsp.viz`, {mod}`pyodsp.model.sp.viz` and {mod}`pyodsp.model.msp.viz`,
each of which imports it lazily inside the plotting call — so a run without the
extra installed works right up until you ask for a chart.

## Solvers

The stage problems are solved through Pyomo, so any solver Pyomo can drive is
in principle usable. In practice:

[HiGHS](https://highs.dev/)
: The default, as `appsi_highs`. Installed by `pip install -e .` through
  `highspy`. This is the only solver the test suite covers.

[Ipopt](https://github.com/coin-or/Ipopt)
: Needed for the two places a quadratic problem shows up: the proximal bundle
  master of dual decomposition — `DdAlgRootBm(..., mode="proximal")`, as in
  `examples/dd/equality_pbm.py` — and the cut-generation master of Benders
  with scaled cuts. `StochasticProgram` reaches for it by default when
  it switches to BDSC; see `cut_master_solver` in
  {doc}`guide/choosing-a-method`. It is not on PyPI as a wheel; install it
  through conda-forge or your package manager.

Other solvers are not tested yet. To use one:

```python
sp = StochasticProgram(
    "model",
    solver="gurobi",
    solver_options={"TimeLimit": 600},
)
```

`solver_options` is passed through to the Pyomo solver call as keyword
arguments.

## MPI (optional)

Every algorithm has an MPI runner that spreads the subproblems across ranks;
see {doc}`guide/mpi`. It needs:

- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
- [MPICH](https://www.mpich.org/) — OpenMPI is not tested

```bash
pip install mpi4py
```

mpi4py is imported only by the `run_mpi` modules, so a serial installation
never touches it.

## Verifying the installation

```bash
pytest                                    # the test suite
python examples/farmer/sp_pipeline.py     # a two-stage run, asserts the known optimum
python examples/inventory/msp_pipeline.py # a multistage run
```

The test suite has `unit`, `integration`, `examples`, `solver` and `mpi`
directories; the last needs `mpiexec` on the path.

# pyodsp
Pyomo interface for Decomposition of Structured Programs, inspired by [DSP](https://github.com/Argonne-National-Laboratory/DSP)

Pyodsp decomposes structured programs written in [Pyomo](https://www.pyomo.org/) and solves the pieces with distributed algorithms — Benders, Benders with scaled cuts, dual decomposition and SDDP — in serial or across MPI ranks. Stochastic programming is the application built on top of that: a front-end that turns a model you write once per stage into the decomposition.

Documentation is available [here](https://hideakiv.github.io/pyodsp/).

## Installation
```bash
pip install -e .
pip install -e ".[viz]"   # adds matplotlib, needed only for the plotting helpers
```

## Features

### Decomposition algorithms (`pyodsp.dec`)
This is the core of pyodsp. A decomposition is a graph of nodes, each owning a Pyomo model and an algorithm object, driven by a runner; you assemble the graph yourself. See the examples.
- **Benders decomposition (bd)** — decomposition on complicating variables. Handles mixed integer linear programs in both stages through van der Laan and Romeijnders' scaled cuts (`bdsc`), implementing:
  > van der Laan, N., & Romeijnders, W. (2024). A converging Benders' decomposition algorithm for two-stage mixed-integer recourse models. *Operations Research*, 72(5), 2190–2214.
- **Dual decomposition (dd)** — decomposition on complicating constraints; the dual of Dantzig–Wolfe decomposition. Accepts MILPs, but solves the Lagrangian dual, so the solution is not guaranteed optimal and the incumbent comes from a heuristic.
- **SDDP** — stochastic dual dynamic programming over a lattice of stages. For the method and its variants, see:
  > Füllner, C., & Rebennack, S. (2025). Stochastic dual dynamic programming and its variants: A review. *SIAM Review*, 67(3), 415–539.

### MPI (optional)
Every algorithm has an MPI runner under `pyodsp.dec` (`run_mpi.py`), which splits the subproblems across ranks. `MultistageProgram(mpi=True)` exposes this for SDDP, where each rank helps evaluate the Monte Carlo sample and rank 0 drives the iteration and owns the results. Launch with `mpiexec -n <ranks> python your_script.py`.

### Stochastic programming front-end (`pyodsp.model`)
Stochastic programming is one application of the decomposition layer: scenarios give the block structure, and the front-end builds the node graph for you from a model you write once per stage — replicating the first-stage variables, keeping the coupling lists aligned, converting sense and results, finding subproblem bounds, and picking the algorithm.
- `StochasticProgram` — two-stage programs. `method` selects `'bd'` (Benders), `'bdsc'` (Benders with scaled cuts), `'dd'` (dual decomposition), or `'de'` to skip decomposition and hand the solver the deterministic equivalent. `'bd'` adapts: an integer recourse either moves to scaled cuts or has its integrality relaxed, with a warning, rather than silently returning a wrong answer.
- `MultistageProgram` — multistage programs over a scenario lattice, solved by SDDP. Uncertainty may be stage-wise independent, stage-varying, Markov, or a lattice you construct yourself.
- Scenarios come from a `ScenarioSet`, a pandas DataFrame, a mapping, or an iterable of dicts; probabilities default to uniform.
- Results (`SpResult`, `MspResult`) carry the first-stage decision, per-scenario outcomes, bound and gap, and the convergence history, all in your own objective's units — with `summary()`, DataFrame accessors, and plots. An SDDP run leaves a policy, so `result.simulate(path)` replays any scenario path against the cuts.

### Risk measures
- `CVaR(alpha=..., weight=...)` optimizes `(1 - weight) * E[Z] + weight * CVaR_alpha[Z]` instead of the expectation, so the solution pays something on average to protect the bad tail. Supported by `method='bd'` and `method='de'`; the other methods cannot see the spread across scenarios and refuse it.

### Analysis
- `sp.analyze()` computes EVPI (the expected value of perfect information) and VSS (the value of the stochastic solution), with `summary()`, `to_frame()` and `plot()`.

## Prerequisites
### Solvers
- [HiGHS](https://highs.dev/)
- Other solvers are not tested yet.

Additionally, the following may be requrired for some of the algorithms.
- [Ipopt](https://github.com/coin-or/Ipopt) (for dual decomposition's proximal bundle master, `DdAlgRootBm(..., mode="proximal")`, and for BDSC's cut-generation master, which is also a proximal bundle method)

### MPI (Optional)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
- [MPICH](https://www.mpich.org/)
- OpenMPI is not tested.

## Examples
Under `examples/`:

| Example | What it shows |
| --- | --- |
| `farmer/sp_pipeline.py` | Birge and Louveaux's farmer through `StochasticProgram`; `--method` switches between bd, dd, bdsc and de |
| `farmer/bd_model.py` | the same problem wired to `pyodsp.dec` by hand, for comparison |
| `inventory/msp_pipeline.py` | multistage inventory through `MultistageProgram`, with `msp_pipeline_mpi.py` for the MPI version |
| `hydro/msp_time_dependent.py` | multistage with stage-varying uncertainty |
| `sslp`, `uc`, `mcsp` | dual decomposition on stochastic server location, unit commitment and multi-commodity problems |
| `bd`, `bdsc`, `dd` | the algorithms driven directly, in serial and MPI form |
| `aircon`, `balance`, `flowergirl` | SDDP and Benders on smaller hand-wired models |

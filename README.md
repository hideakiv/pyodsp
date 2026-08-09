# pyodsp
Pyomo interface for Decomposition of Structured Programs, inspired by [DSP](https://github.com/Argonne-National-Laboratory/DSP)

Pyodsp offers distributed algorithms for programming models written in [Pyomo](https://www.pyomo.org/), and a stochastic programming front-end that puts them behind a model you write once per stage.

## Installation
```bash
pip install -e .
pip install -e ".[viz]"   # adds matplotlib, needed only for the plotting helpers
```

## Documentation
Full documentation lives in `docs/` — a guide to both front-ends, how the
algorithm is chosen, risk measures, EVPI/VSS, MPI, and an API reference
generated from the docstrings.

```bash
pip install -e ".[docs]"
make -C docs html      # then open docs/_build/html/index.html
make -C docs strict    # what CI should run: warnings are errors
```

## Quick start
A two-stage stochastic program: describe the first stage, describe one scenario's recourse, hand over the scenarios.

```python
import pyomo.environ as pyo
from pyodsp.model.sp import StochasticProgram

sp = StochasticProgram("farmer", sense="max")

@sp.first_stage
def first_stage(m):
    m.acres = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
    m.land = pyo.Constraint(expr=sum(m.acres[c] for c in CROPS) <= 500)
    return -sum(COST[c] * m.acres[c] for c in CROPS)

@sp.recourse
def recourse(m, state, scenario):
    m.sold = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
    m.balance = pyo.Constraint(
        CROPS,
        rule=lambda m, c: m.sold[c] == scenario["yield"][c] * state.acres[c],
    )
    return sum(PRICE[c] * m.sold[c] for c in CROPS)

sp.set_scenarios({name: {"yield": y} for name, y in YIELDS.items()})
result = sp.solve()
print(result.summary())
```

`state.acres` is the scenario's own copy of the first-stage decision. The pipeline replicates the first-stage variables into each scenario, keeps the coupling lists aligned, converts a maximize program to the minimize form the algorithms require and converts every result back, finds a valid bound per subproblem, and picks the algorithm. `sp.describe()` reports what it decided; `sp.build()` does all of that without solving.

Multistage problems are stated the same way, one builder for every stage, and solved by SDDP:

```python
from pyodsp.model.msp import MultistageProgram

msp = MultistageProgram("inventory", sense="min", stage_bound=0.0)

@msp.stage(state=["inventory"])
def stage(m, state, node):
    m.inventory = pyo.Var(bounds=(0, 100))      # what this stage passes on
    m.buy = pyo.Var(bounds=(0, 50))
    m.balance = pyo.Constraint(
        expr=m.inventory == state.inventory + m.buy - node["demand"]
    )
    return COST * m.buy

msp.set_initial_state(inventory=20.0)
msp.set_realizations(realizations, stages=4)
result = msp.solve()
```

`state.inventory` is what the previous stage left, `m.inventory` what this one leaves — a variable at every stage but the first, where it is the initial condition, which is why one builder covers the whole horizon.

## Features

### Modeling front-end (`pyodsp.model`)
- `StochasticProgram` — two-stage programs. `method` selects `'bd'` (Benders), `'bdsc'` (Benders with scaled cuts), `'dd'` (dual decomposition), or `'de'` to skip decomposition and hand the solver the deterministic equivalent. `'bd'` adapts: an integer recourse either moves to scaled cuts or has its integrality relaxed, with a warning, rather than silently returning a wrong answer.
- `MultistageProgram` — multistage programs over a scenario lattice, solved by SDDP. Uncertainty may be stage-wise independent, stage-varying, Markov, or a lattice you construct yourself.
- Scenarios come from a `ScenarioSet`, a pandas DataFrame, a mapping, or an iterable of dicts; probabilities default to uniform.
- Results (`SpResult`, `MspResult`) carry the first-stage decision, per-scenario outcomes, bound and gap, and the convergence history, all in your own objective's units — with `summary()`, DataFrame accessors, and plots. An SDDP run leaves a policy, so `result.simulate(path)` replays any scenario path against the cuts.

### Risk measures
- `CVaR(alpha=..., weight=...)` optimizes `(1 - weight) * E[Z] + weight * CVaR_alpha[Z]` instead of the expectation, so the solution pays something on average to protect the bad tail. Supported by `method='bd'` and `method='de'`; the other methods cannot see the spread across scenarios and refuse it.

### Analysis
- `sp.analyze()` computes EVPI (the expected value of perfect information) and VSS (the value of the stochastic solution), with `summary()`, `to_frame()` and `plot()`.

### Decomposition algorithms (`pyodsp.dec`)
The algorithms are usable directly when you want to wire the node graph yourself; see the examples.
- **Benders decomposition (bd)** — decomposition on complicating variables. Handles mixed integer linear programs in both stages through van der Laan and Romeijnders' scaled cuts (`bdsc`), implementing:
  > van der Laan, N., & Romeijnders, W. (2024). A converging Benders' decomposition algorithm for two-stage mixed-integer recourse models. *Operations Research*, 72(5), 2190–2214.
- **Dual decomposition (dd)** — decomposition on complicating constraints; the dual of Dantzig–Wolfe decomposition. Accepts MILPs, but solves the Lagrangian dual, so the solution is not guaranteed optimal and the incumbent comes from a heuristic.
- **SDDP** — stochastic dual dynamic programming over a lattice of stages. For the method and its variants, see:
  > Füllner, C., & Rebennack, S. (2025). Stochastic dual dynamic programming and its variants: A review. *SIAM Review*, 67(3), 415–539.

### MPI (optional)
Every algorithm has an MPI runner under `pyodsp.dec` (`run_mpi.py`), which splits the subproblems across ranks. `MultistageProgram(mpi=True)` exposes this for SDDP, where each rank helps evaluate the Monte Carlo sample and rank 0 drives the iteration and owns the results. Launch with `mpiexec -n <ranks> python your_script.py`.

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

# Examples

Everything under `examples/` runs standalone. Most take `--solver` and assert a
known optimum at the end, so a run that finishes silently is a run that got the
right answer.

```bash
python examples/farmer/sp_pipeline.py
python examples/farmer/sp_pipeline.py --method dd --plot
python examples/inventory/msp_pipeline.py --stages 6
```

Several import a sibling module by bare name (`from optimality import ...`), so
run them from inside their own directory, or with that directory on
`PYTHONPATH`.

## Through the modelling front-end

These are the ones to read first.

`farmer/sp_pipeline.py`
: Birge and Louveaux's farmer — the canonical two-stage example — through
  {class}`~pyodsp.model.sp.problem.StochasticProgram`. `--method` switches between
  `bd`, `dd`, `bdsc` and `de`, all of which reach the same 108,390.

`inventory/msp_pipeline.py`
: A multistage inventory problem through
  {class}`~pyodsp.model.msp.problem.MultistageProgram`: buy early and cheap before you
  know the demand, or late and dear once you do. Ends by replaying the
  high-demand path through the policy. `msp_pipeline_mpi.py` is the same model
  with `mpi=True`.

`hydro/msp_time_dependent.py`
: Hydro reservoir scheduling, and the best example of how to handle time
  dependence. All three kinds appear in the one model: deterministic data that
  varies by stage (the electricity price, indexed by `node.stage`), a
  distribution that varies by stage (wetter inflows in spring —
  `set_stage_realizations`), and a process that depends on its own past
  (autocorrelated inflows).

## The same problem, both ways

`farmer/bd_model.py`
: The farmer wired directly to {mod}`pyodsp.dec`. Read against
  `sp_pipeline.py`: here the first-stage variables are re-declared inside every
  scenario, the two coupling lists are built in matching order by hand, each
  leaf is given an invented bound, and the maximize objective is negated on
  every model before the algorithms will accept it. None of that appears in the
  pipeline version. `model.py` and `stochastic_model.py` hold the shared data.

`balance/sddp.py`, `aircon/sddp.py`
: Hand-wired multistage runs, against `inventory/msp_pipeline.py` for the
  front-end version of the same idea. `aircon/sddp_mpi.py` is the MPI variant;
  `aircon/bd.py` solves the same model by Benders instead.

## Driving the algorithms directly

`bd/optimality.py`, `bd/feasibility.py`
: The smallest complete Benders runs — a two-variable master and two
  scenarios. `optimality.py` is the one quoted in {doc}`guide/low-level`;
  `feasibility.py` exercises feasibility cuts. `optimality_mpi.py` distributes
  the same three nodes over three ranks.

`dd/equality*.py`
: Dual decomposition, in five variants: the base LP (`equality.py`), a MILP
  (`equality_mip.py`), the proximal bundle master that needs Ipopt
  (`equality_pbm.py`), and MPI versions of the first two.

`dd/ray.py`
: Dual decomposition where the subproblems are unbounded, so the master is fed
  extreme rays rather than points. Also the clearest example that DD is not
  only for scenarios: the two children here are different blocks of one
  deterministic problem, tied by a single complicating constraint.

`bdsc/cs.py`
: Benders with scaled cuts on the Carøe and Schultz (1999) instance — a
  two-stage problem with integer recourse, which is what BDSC exists for.
  `cs_mpi.py` distributes it.

## Larger models

`sslp/`
: Stochastic server location (Ntaimo and Sen, 2005), by dual decomposition. The
  scenario subproblems keep `Binary` replicas of the first-stage variables,
  which is what DD's non-anticipativity constraints need — a good illustration
  of why the replica domain differs from Benders'.

`uc/`
: Unit commitment by dual decomposition, with a custom MIP heuristic in
  `heuristics.py` for recovering a primal solution from the Lagrangian bound.

`mcsp/`
: The multiple cutting-stock problem (Belov and Scheithauer, 2002) by dual
  decomposition. `dp_*.py` are a dynamic-programming subproblem variant.

`flowergirl/bd.py`
: A small multistage newsvendor, by Benders.

`balance/`
: An energy balancing model with a regime-switching scenario generator —
  `scenarios.py` and `regime.py` build the lattice, `run.py` drives it, `lp.py`
  is the extensive form for comparison.

## What the tests run

`test/examples/` runs a subset of these as integration tests, so they are kept
working. `test/mpi/` covers the MPI runners and needs `mpiexec` on the path.

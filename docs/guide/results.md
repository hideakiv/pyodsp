# Reading results

Everything a run returns is in **your** objective's units. The algorithms work
on a converted minimize form, and each solver remembers whether it flipped, so
the conversion back happens on the way out — you never apply a sign yourself.

## Two-stage: `SpResult`

```python
result = sp.solve()
```

```{list-table}
:header-rows: 1
:widths: 30 70

* - Attribute
  - Meaning
* - `objective`
  - The value of the returned solution **under the program's risk measure** —
    what was actually optimized. Equal to `expected_objective` when the program
    is risk-neutral. `None` if no solution was recovered.
* - `expected_objective`
  - The probability-weighted average objective, whatever the risk measure.
* - `bound`
  - The algorithm's final bound on the optimum.
* - `gap`
  - Relative distance between the two. `None` if either is missing.
* - `first_stage`
  - The here-and-now decision, nested by variable name: a float for a scalar
    variable, `{index: value}` for an indexed one.
* - `first_stage_flat`
  - The same, flattened to `{"acreage[WHEAT]": 170.0, ...}`.
* - `scenarios`
  - One {class}`~pyodsp.model.sp.result.ScenarioOutcome` per scenario, in scenario
    order.
* - `history`
  - A DataFrame of per-iteration `bound` and `incumbent`.
* - `method`
  - Which algorithm actually ran — not necessarily the one you asked for.
* - `first_stage_consistent`
  - Whether every scenario agrees on the first-stage decision.
* - `output_dir`
  - Where the per-node files were written.
```

`summary()` prints the lot; `first_stage_frame()` and `scenario_frame()` give
DataFrames.

### A trap worth knowing

`ScenarioOutcome.objective` is **that scenario model's own objective**, and
what that covers depends on the algorithm:

- Under Benders and BDSC it is the recourse objective *alone*, because the
  first stage lives in the master. In the farmer example the scenario values
  come back in the hundreds of thousands while the answer is 108,390 — those
  are sales revenue before planting costs, not per-scenario profits.
- Under dual decomposition each scenario carries a complete,
  probability-weighted copy of the problem, so it is that scenario's weighted
  share of the total instead.

Summing them is meaningful in the second case and not in the first. Use
`result.objective` for the answer.

### `first_stage_consistent`

Always true under Benders, where a single master holds the first stage. Under
dual decomposition it is real information: it says whether non-anticipativity
was actually attained, or whether the run stopped with the scenarios still
disagreeing about what to do here and now. `summary()` prints a warning line
when it is false.

## Multistage: `MspResult`

```python
result = msp.solve()
```

The shape differs because SDDP's answer differs. There is no incumbent
sequence: the method approaches the optimum from one side and estimates the
other by simulation.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Attribute
  - Meaning
* - `bound`
  - The root's final bound. **Not** a proven optimum.
* - `first_stage`, `first_stage_flat`
  - The here-and-now decision, as in the two-stage case.
* - `history`
  - The root master's per-iteration trajectory.
* - `simulation`
  - One row per convergence test: the bound against a confidence interval for
    the simulated policy. This is what stands in for an incumbent.
* - `simulation_samples`
  - The individual objective values behind the last such test.
* - `num_stages`, `nodes_per_stage`, `lattice`
  - The scenario structure the run was built on.
* - `is_root_rank`
  - Whether this result carries the answer. Always true without MPI.
```

`objective_stats()` summarizes the simulated sample — mean, spread, quantiles,
plus the bound and the last interval. The policy is chosen on its expectation;
this is the sample that expectation was taken over, so it also says how wide
and how skewed the outcomes behind it are.

### Replaying a path

```python
for step in result.simulate(["0-0", "1-2", "2-2"]):
    step.node_idx, step.stage_cost, step.next_state
```

`result.policy()` returns the {class}`~pyodsp.dec.sddp.policy.SddpPolicy` if
you want to hold on to it and simulate repeatedly.

## Plots

All plotting needs matplotlib (`pip install -e ".[viz]"`) and is imported
lazily, so a run without it works until you ask for a chart. Every `plot_*`
method writes into the run's output directory unless given a path, and returns
the path written.

```python
result.plot()                         # two-stage: every applicable chart
result.plot_convergence()             # multistage
result.plot_objective_distribution()
result.plot_scenario_lattice()
result.plot_state_trajectory(paths, "inventory")
result.plot(paths)                    # multistage: everything, given paths to draw

analysis.plot()                       # the EVPI/VSS chart
```

A multistage policy has no single trajectory of its own, which is why
`MspResult.plot` requires you to name the scenario paths to draw.

Charts take `theme="light"` or `theme="dark"`.

## On disk

Each node writes into `output/<name>/` (or your `output_dir`). The convergence
history the result objects expose is read back from there, so a finished run
stays inspectable — and a rerun of the same scenario set writes to the same
places, because scenario order fixes the leaf indices.

## Reference

{class}`~pyodsp.model.sp.result.SpResult`,
{class}`~pyodsp.model.sp.result.ScenarioOutcome` and
{class}`~pyodsp.model.msp.result.MspResult` are documented in {doc}`../api/model`.

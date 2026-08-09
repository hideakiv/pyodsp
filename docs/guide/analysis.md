# EVPI and VSS

Two questions worth asking about any stochastic program: what would it be worth
to know the future, and was modelling the uncertainty worth the trouble?
Neither is a problem you solve — each is a comparison between problems.

```python
analysis = sp.analyze()
print(analysis.summary())
analysis.plot()
```

```text
farmer: value of information
  WS  (perfect information) : 115,406
  RP  (stochastic program)  : 108,390
  EEV (mean-value decision) : 107,240

  EVPI = RP - WS   : 7,015.56  (6.47% of RP)
  VSS  = EEV - RP  : 1,150  (1.06% of RP)
```

## The three reference points

Following Birge and Louveaux, ch. 4:

**WS** — wait-and-see
: Each scenario solved as if it had been known in advance, averaged. The cost
  of a decision maker who sees the future. One solve per scenario.

**RP** — the recourse problem
: The stochastic program itself. Your actual answer.

**EEV** — expected result of the expected-value solution
: Solve the mean-value problem — one scenario, the data averaged — then fix
  that first-stage decision and evaluate it against every real scenario. The
  cost of pretending the mean is the truth. Two extra solves.

For a minimization $\mathrm{WS} \le \mathrm{RP} \le \mathrm{EEV}$, and the two
gaps are

$$\mathrm{EVPI} = \mathrm{RP} - \mathrm{WS}, \qquad
  \mathrm{VSS} = \mathrm{EEV} - \mathrm{RP}.$$

Maximization reverses both orderings, so the differences are signed by the
sense and both measures come out non-negative either way.

## Reading them

**EVPI** is what perfect information would be worth. Zero means the same
decision is optimal however the uncertainty resolves — knowing the future would
buy you nothing, and the uncertainty, whatever its spread, does not bear on
this decision.

**VSS** is what modelling the uncertainty was worth. Zero means planning
against the mean happens to give the same decision, so a deterministic model
would have done. It is never negative: the stochastic solution is optimal over
the same feasible set the mean-value decision came from.

A large VSS is the argument for having built the stochastic model at all. A
large EVPI is an argument for spending money on forecasting.

## The infeasible case

Sometimes the mean-value decision cannot be carried out at all in some
scenario — the plan built for average demand leaves no way to meet high demand.
Then EEV is not merely large, it does not exist, and VSS is unbounded:

```python
if analysis.eev_infeasible:
    ...   # analysis.eev is None, analysis.vss is None
```

`summary()` says so explicitly. This is the strongest possible statement that
the uncertainty needed modelling.

## Cost, and skipping half of it

`analyze()` solves the program itself, then one deterministic-equivalent
problem per scenario for EVPI and two more for VSS. On a large scenario set
that is the dominant cost, so ask only for what you need:

```python
analysis = sp.analyze(measures=["vss"])     # skips the per-scenario solves
```

The reference problems are built from the same two builder functions as the
program they come from — same model, only the scenarios differ — and each is
solved as a deterministic equivalent, whatever method the parent program uses.
They are written into subdirectories of the parent's output directory.

## When the mean is not meaningful

The expected-value problem averages the scenario data. That works for numbers
and falls apart for anything else — a categorical regime label, a set of
active constraints, an integer count where the mean is fractional. Override the
entries that cannot be averaged:

```python
analysis = sp.analyze(mean_scenario_data={"regime": "normal"})
```

Only the named entries are replaced; the rest are averaged as usual.

## What comes back

{class}`~pyodsp.model.sp.analysis.SpAnalysis` keeps the solves it made, so the
mean-value decision itself is reachable:

```python
analysis.rp                  # SpResult for the stochastic program
analysis.ev                  # SpResult for the mean-value problem
analysis.ev.first_stage      # the decision a deterministic model would have made
analysis.scenario_values     # each scenario's wait-and-see optimum, by name
analysis.to_frame()          # WS, RP, EEV, EVPI, VSS as a DataFrame
analysis.relative_evpi       # as a fraction of RP
```

To compare several programs side by side:

```python
from pyodsp.model.sp.analysis import measures_frame
measures_frame([analysis_a, analysis_b, analysis_c])
```

## Reference

{class}`~pyodsp.model.sp.analysis.SpAnalysis` and
{func}`~pyodsp.model.sp.analysis.analyze` are documented in
{doc}`../api/model`.

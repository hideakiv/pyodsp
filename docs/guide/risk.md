# Risk measures

By default a stochastic program minimizes the *average* cost. That is what
"stochastic programming" means unqualified, and it is the right objective when
you will face the same problem many times. It is the wrong one when a single
bad outcome is what you actually care about.

A risk measure changes the problem, not the way it is solved.

```python
from pyodsp.model.sp import StochasticProgram, CVaR

sp = StochasticProgram("model", risk=CVaR(alpha=0.95, weight=0.5))
```

## CVaR

{class}`~pyodsp.alg.risk.CVaR` — Conditional Value at Risk, also called
expected shortfall — is the mean of the worst outcomes. Blended with the
expectation, the objective becomes

$$(1 - w)\,\mathbb{E}[Z] \;+\; w\,\mathrm{CVaR}_\alpha[Z]$$

where $\mathrm{CVaR}_\alpha[Z]$ averages over the worst $1 - \alpha$ of
outcomes.

`alpha`
: The confidence level, in $[0, 1)$. `0.95` attends to the worst 5%. At `0` it
  is the plain expectation; approaching `1` it becomes the single worst
  outcome. Exactly `1` is rejected — the tail is empty there and CVaR is
  undefined — so use a value just below it.

`weight`
: How much of the objective the tail accounts for, in $[0, 1]$. `0` is exactly
  risk-neutral, and worth running as a check: it must reproduce the expectation
  solution.

The solution you get will pay something in expectation to protect against the
outcomes that hurt. Comparing `result.objective` (risk-adjusted, what was
actually optimized) against `result.expected_objective` (the plain
probability-weighted average) tells you how much:

```python
result = sp.solve()
print(result.objective)            # under the risk measure
print(result.expected_objective)   # what it cost you in expectation
```

`result.summary()` prints both when the program is risk-averse.

## Both senses, one formulation

Everything in {mod}`pyodsp.alg.risk` is stated for a **minimization**: larger
is worse, and the tail of interest is the upper one. That is the only
convention the algorithms ever see, since a maximize model is converted on
construction.

The consequence is the one you want: for a maximize program, the upper tail of
the converted objective is the lower tail of the original — the bad outcomes.
So `CVaR(alpha=0.95)` protects against low profit in a maximization and against
high cost in a minimization, with no change on your side.

## Which methods accept one

Only `method='bd'` and `method='de'`. See {doc}`choosing-a-method` for the full
reasoning; briefly, a risk measure prices the *spread* across scenarios, which
needs each scenario's cost visible to the master separately. Benders keeps one
theta per scenario. BDSC aggregates them into a single cut and dual
decomposition never compares them, so neither can state the tail, and both
raise rather than quietly optimizing the expectation.

This has one awkward consequence. Integer recourse normally sends `'bd'` to
BDSC — but BDSC cannot carry the risk measure, so the combination raises and
asks you to choose:

```python
# integer recourse + CVaR: pick one of these
sp = StochasticProgram(..., risk=CVaR(), integer_recourse="relax")  # LP relaxation
sp = StochasticProgram(..., risk=CVaR(), method="de")               # extensive form, exact
```

Multistage programs are risk-neutral: {class}`~pyodsp.model.msp.problem.MultistageProgram`
takes no `risk` argument.

## Measuring a sample yourself

The incumbent has to be measured with the same yardstick as the bound, or the
two never meet. That is what {func}`~pyodsp.alg.risk.cvar_of_sample` is for —
it computes CVaR of a discrete distribution exactly, walking the outcomes
worst-first and splitting the one that straddles the tail boundary, which is
the value the master's Rockafellar–Uryasev program converges to.

```python
from pyodsp.alg.risk import cvar_of_sample, value_of_sample

cvar_of_sample(values, probabilities, alpha=0.95)
value_of_sample(risk, values, probabilities)   # whatever the measure is
```

Both take values in the minimize convention.

## Reference

{class}`~pyodsp.alg.risk.CVaR`, {class}`~pyodsp.alg.risk.Expectation` and the
sample helpers are documented in {doc}`../api/support`.

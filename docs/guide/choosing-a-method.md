# Choosing a method

`StochasticProgram(method=...)` takes one of four values. This page is about
which one to use, what each one silently requires, and what the pipeline does
when you ask for something that cannot work.

```python
sp = StochasticProgram("model", method="bd")   # the default
```

## The four

```{list-table}
:header-rows: 1
:widths: 12 30 30 28

* - `method`
  - Decomposes on
  - Handles
  - Gives you
* - `'bd'`
  - complicating **variables** — the first stage stays in a master, each
    scenario becomes a subproblem priced by a cut
  - continuous recourse; integer recourse only by adapting (below)
  - a proven optimum, with a bound and gap that close
* - `'bdsc'`
  - the same, but cuts are built to recover the convex hull of the recourse objective using row generation
  - integer recourse, exactly
  - a proven optimum; needs a quadratic-capable solver for its inner master
* - `'dd'`
  - complicating **constraints** — each scenario carries a whole copy of the
    problem, tied together by non-anticipativity
  - MILPs anywhere
  - the Lagrangian dual bound, plus a heuristic incumbent
* - `'de'`
  - nothing; the extensive form goes to the solver in one piece
  - anything the solver handles
  - whatever the solver proves
```

## The short version

Start with `'bd'`. It is the default because it is the right answer for the
common case: a two-stage program with a continuous second stage, where Benders
is both fast and exact.

Reach for something else when:

- **The model is small, or you are checking a decomposition against the truth.**
  Use `'de'`. It solves the deterministic equivalent directly — no master, no
  subproblems, no iteration — which is the ground truth every other method
  should reproduce. It is also the only method besides `'bd'` that accepts a
  risk measure. It stops being viable when the extensive form stops fitting in
  memory, which is the whole reason decomposition exists.
- **The second stage has integer variables and you need them exact.** Use
  `'bdsc'`, or let `'bd'` switch to it for you.
- **The coupling is a shared constraint rather than a shared variable**, or the
  problem is a large MILP where a bound is enough. Use `'dd'`.

## When `'bd'` adapts

Benders cuts are built from the LP duals of the subproblems. An integer second
stage does not have those. So `'bd'` is the one method that does not take your
word for it: `build()` scans for integer variables, and if any are in the
recourse it either changes the algorithm or changes the problem — and warns
either way. It never silently stays on plain Benders, which would return a
confidently wrong answer.

Which of the two happens is `integer_recourse`:

`integer_recourse='bdsc'` (the default)
: Switch to Benders with scaled cuts, which handles integer recourse exactly.
  Costs you a dependency: BDSC's cut-generation master runs a proximal bundle
  method, so it needs a quadratic-capable solver of its own. That is
  `cut_master_solver`, `'ipopt'` by default. The algorithm is
  {ref}`van der Laan and Romeijnders (2024) <bdsc-citation>`.

`integer_recourse='relax'`
: Relax the integrality of the recourse variables and stay on Benders. Bounds
  are preserved, so a binary becomes a continuous variable on $[0, 1]$ — this
  is the LP relaxation, not a different problem. The result bounds the true
  optimum rather than attaining it, and the second-stage solution may come back
  fractional.

Integer *first-stage* variables are fine under plain Benders and trigger
nothing: the master is a MILP, its subproblems are still LPs, and their duals
still exist.

:::{note}
The scan probes the recourse on the first scenario only, on the reasoning that
integrality is declared by the builder — one function — and so does not vary
with the data in any normal model. `build_bd` re-checks every scenario it
actually constructs, so a builder that *does* branch on its data is caught
there rather than quietly mis-decomposed.
:::

## What each method requires

### A complete state vector — `'bdsc'` and `'dd'`

Benders keeps the first stage in its master, so a first-stage variable the
recourse never reads simply stays there, and `state=[...]` may name a subset.

The other two embed the first stage inside every scenario. A first-stage
variable left out of the state vector is then no longer a first-stage variable
at all:

- **BDSC** prices columns over each scenario's own feasible set, which now
  includes the whole first stage. A variable outside the coupling vector is one
  the scenario may set freely and never report, so the columns describe a
  larger set than the master is restricted to, and the cuts do not bound it.
- **DD** ties the scenarios together with non-anticipativity constraints built
  over the state vector alone. A variable outside it gets an independent copy
  per scenario with nothing equating them — a here-and-now decision quietly
  turned into a wait-and-see one.

Both refuse to build in that case. Drop the `state=[...]` argument to couple
everything, or use `'bd'`.

### A quadratic-capable inner solver — `'bdsc'`

`cut_master_solver='ipopt'` by default; see {doc}`../installation`.

### A risk-capable master — `'bd'` and `'de'` only

A risk measure prices the *spread* across scenarios, so the master has to see
each scenario's cost separately. Benders does — one theta per scenario. BDSC
aggregates them into a single cut, and dual decomposition puts a whole copy of
the problem in each scenario with nothing comparing them. Passing `risk=` to
either raises. See {doc}`risk`.

The awkward corner is integer recourse *and* a risk measure: the default
`integer_recourse='bdsc'` would send the problem to a method that cannot carry
the risk measure, so that combination raises too, and tells you to pick between
`integer_recourse='relax'` and `method='de'`.

## What each method gives back

This is the part worth being careful about, because all four return an
`SpResult` with an `objective` and a `bound` on it, and they do not all mean
the same thing.

**Benders and BDSC** converge from both sides. `result.gap` closing to zero is
a proof of optimality.

**Dual decomposition** solves the Lagrangian dual. For a convex problem that
matches the primal optimum. For a non-convex one — which is to say any MILP —
it sits strictly below it, so:

- `result.bound` is a valid lower bound and nothing more,
- `result.objective` comes from a MIP heuristic run at the end (`heuristic=True`,
  the default; without it you get a bound and no primal solution at all),
- the gap between them need not close, and a run that stops at
  `max_iteration` with a gap open is the expected outcome, not a failure.

`'dd'` warns about exactly this when it finds integer variables. If you need a
proven optimum on that model, use `'bd'`.

**The deterministic equivalent** returns whatever the solver proved. There is
one solve and no iteration, so `result.history` comes back empty and
`result.bound` equals `result.objective` — the optimum was proved on the whole
problem rather than approached. If the solver does not reach optimality the
pipeline raises, because with no decomposition involved the problem is the
model itself.

Dual decomposition also reports `result.first_stage_consistent`: whether the
scenarios actually agreed on the first-stage decision. Under Benders there is
one master and it is always true; under DD it tells you whether
non-anticipativity was attained.

## Overruling the decision

`resolved_method` says what `build()` settled on:

```python
sp.build()
assert sp.resolved_method == "bd"
```

and `describe()` prints it alongside what you asked for, when they differ.
Anything the pipeline decides for you it also warns about, so a run whose
warnings you have read is a run whose algorithm you have agreed to.

## References

(bdsc-citation)=

Benders with scaled cuts — `method='bdsc'`, and where `'bd'` sends an integer
recourse — implements:

> van der Laan, N., & Romeijnders, W. (2024). A converging Benders'
> decomposition algorithm for two-stage mixed-integer recourse models.
> *Operations Research*, 72(5), 2190–2214.

The other algorithms are textbook: Benders decomposition and dual
decomposition as in Birge, J. R., & Louveaux, F. (2011), *Introduction to
Stochastic Programming* (2nd ed.), Springer.

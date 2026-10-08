"""Stochastic dual dynamic programming with scaled cuts.

Plain SDDP builds its cuts from the LP duals of each stage's subproblem,
which an integer stage does not have. This variant applies BDSC's
convex-hull cut construction (see pyodsp.dec.bdsc) at each stage of the
backward pass, so the cost-to-go approximation converges on multistage
mixed-integer recourse models rather than returning a bound that never
closes.

Scope. Benders decomposition with scaled cuts (van der Laan & Romeijnders,
2023) and its multistage generalization (Romeijnders & van der Laan, 2024)
both need the recourse's feasible region as context for the cut: a scaled
cut is built by column generation over a subproblem that embeds a free
copy of the incoming state's feasible set, not just a tangent at one trial
point. In a genuinely multistage model that feasible set depends on the
decisions made to reach it, which the 2024 paper's own hierarchy handles
by letting cuts depend on the whole ancestor path -- a tool it stops short
of turning into a concrete algorithm (left as future work), because
path-dependent cuts aren't shareable across a lattice's nodes the way
SDDP's cost-to-go functions need to be.

This module targets a narrower, SDDP-compatible case: every stage's
transition (its constraint data: T, W, h) is the same across every node at
that stage, and only the cost coefficients are random. A battery schedule
against a stochastic electricity price is the model case -- the level
transition is deterministic, the price is not. Under this restriction a
stage-t leaf's column-generation subproblem can embed the deterministic
transition chain from stage t-K through t-1 (K the lookback, maximum by
default -- i.e. back to stage 1) as ordinary free variables, while
declaring only x_{t-1} as its coupling variable. The resulting scaled cut
is still exactly a function of x_{t-1}, because only coupling variables
can carry a cut coefficient; the earlier stages only sharpen which x_{t-1}
the row generation considers reachable. That keeps the cut Markovian and
shareable across the lattice the way SDDP needs, at the cost of solving a
larger (but still scenario-independent, since the chain itself doesn't
branch) subproblem the deeper the lookback goes.

Concretely: a stage's root is alg_root_bm.SddpScAlgRootBm, new to this
module -- Lattice visits a node once per outer iteration (run_step on the
forward pass, add_cuts separately on the backward pass, no inner loop
between them), so the rho fixed point at the heart of a scaled cut (van
der Laan & Romeijnders, Lemma 2-4) advances by one step per visit rather
than converging within it, refining across SDDP's own many iterations
instead. A stage's leaf is bdsc.alg_leaf_pyomo.BdScAlgLeafPyomo, reused
unchanged (see alg_leaf_pyomo.py) -- its row-generation machinery doesn't
need to know it's embedded in a lattice rather than a two-stage master.

For the two-stage paper's method itself, see pyodsp.dec.bdsc.
"""

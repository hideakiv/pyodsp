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

Status. Validated end to end for a root+leaf (two-stage) lattice: driving
the Caroe-Schultz instance from examples/bdsc/cs.py through Lattice instead
of bdsc's HubAndSpoke converges to the same bound.

An inner stage (an INodeInner between the root and the leaves) is an
SddpScNodeInner (node.py) whose root role and leaf role use separate Pyomo
models -- sharing one, as plain Benders inner nodes do, does not fit: both
roles run their own cut-adding bundle method with its own Objective, and
the leaf role's column generation needs the incoming state free while the
root role needs it fixed. The node carries what sharing would have given
for free, in both directions: the incoming state down into the root role's
mutable Params (SddpScAlgRootBm.state_params / set_state), and the root
role's cuts -- the next stage's cost-to-go -- into the leaf role's future
term (SddpScAlgLeafPyomo.future_vars / set_future_cuts), so the cut an
inner stage reports covers everything downstream. On a 3-stage toy model
with a hand-verifiable optimum the bound converges to it (it converged to
the value ignoring the last stage before the future term existed; see
test/solver/test_sddpsc_multistage.py).

Known gaps: an inner node's root role does not yet see its ancestors' cost
(the parent objective is set on the leaf role's model only), so simulated
path costs -- the statistical upper bound -- leave it out. And with more
than one node per stage, every node's rho is advanced from c's that the
children computed against whichever node was visited last.

For the two-stage paper's method itself, see pyodsp.dec.bdsc.
"""

"""Benders decomposition with scaled cuts.

Plain Benders builds its cuts from the LP duals of the subproblems, which
an integer second stage does not have. This variant creates the cuts that 
recover the convex hull of the recourse objective using row generation, 
so it converges on a two-stage mixed-integer recourse model rather than
returning a bound that never closes. The cut-generation master is solved
by row generation, as in the paper's Algorithm 2: an LP over the cut
coefficients, solved with the subproblem's own solver. The earlier proximal
bundle master (cut_master='pbm', quadratic) is kept for comparison.

Implements:

    van der Laan, N., & Romeijnders, W. (2024). A converging Benders'
    decomposition algorithm for two-stage mixed-integer recourse models.
    Operations Research, 72(5), 2190-2214.
"""

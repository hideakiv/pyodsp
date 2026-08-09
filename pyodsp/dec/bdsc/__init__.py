"""Benders decomposition with scaled cuts.

Plain Benders builds its cuts from the LP duals of the subproblems, which
an integer second stage does not have. This variant creates the cuts that 
recover the convex hull of the recourse objective using row generation, 
so it converges on a two-stage mixed-integer recourse model rather than
returning a bound that never closes. The cut generation master uses 
proximal bundle method, so it needs a quadratic-capable solver.

Implements:

    van der Laan, N., & Romeijnders, W. (2024). A converging Benders'
    decomposition algorithm for two-stage mixed-integer recourse models.
    Operations Research, 72(5), 2190-2214.
"""

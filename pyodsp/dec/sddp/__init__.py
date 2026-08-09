"""Stochastic dual dynamic programming.

Multistage decomposition over a recombining lattice. Each stage keeps a
cut approximation of its cost-to-go: a forward pass samples a scenario
path and solves it stage by stage against the cuts it has, and a backward
pass refines them from the duals it collected on the way. The root
master's objective is a bound on the optimum, while the other side is
estimated by simulating the policy, so a run converges on an interval
rather than on a proven gap.

For a survey of the method and the variants it has grown, see:

    Füllner, C., & Rebennack, S. (2025). Stochastic dual dynamic
    programming and its variants: A review. SIAM Review, 67(3), 415-539.
"""

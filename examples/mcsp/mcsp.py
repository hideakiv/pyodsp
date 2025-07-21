"""
Multiple Cutting Stock Problem
Belov, G., & Scheithauer, G. (2002). A cutting plane algorithm for the one-dimensional cutting stock problem with multiple stock lengths. European Journal of Operational Research, 141(2), 274-294.

Sets:
    K: number of types of rolls
    N_k: number of rolls of type k
    P: number of pieces

Parameters
    L_k: length of roll of type k
    c_k: cost of roll of type k
    d_p: demand of piece p
    l_p: length of piece p

Variables:
    x_{k,n,p}: number of piece p in the nth roll of type k
    xtot_{k,p}: number of piece p in rolls of type k
    y_{k,n}: indicator of the usage of the nth roll of type k

min_{x,y}   sum_{k=1}^{K}sum_{n=1}^{N_k} c_k y_{k,n}
s.t.        sum_{k=1}^{K} xtot_{k,p} >= d_p, forall p = 1,...,P
            sum_{n=1}^{N_k} x_{k,n,p} == xtot_{k,p}, forall n = 1,...,N_k, p = 1,...,P
            sum_{p=1}^{P} l_p x_{k,n,p} <= L_k y_{k,n}, forall n = 1,...,N_k, k = 1,...,K
            y_{k,n} in {0,1}, forall n = 1,...,N_k, k = 1,...,K
            x_{k,n,p} in Z_+, forall n = 1,...,N_k, k = 1,...,K, p = 1,...,P
"""

import pyomo.environ as pyo


def master_problem(model: pyo.ConcreteModel, N: list[int], P: int, d: list[int]):
    K = len(N)

    model.xtot = pyo.Var(range(K), range(P), domain=pyo.NonNegativeIntegers)

    def rule_demand(model, p):
        return sum(model.xtot[k, p] for k in range(K)) >= d[p]

    model.demand = pyo.Constraint(range(P), rule=rule_demand)


def sub_problem(model: pyo.Block, N: int, P: int, L: int, c: float, l: list[int]):
    model.x = pyo.Var(range(N), range(P), domain=pyo.NonNegativeIntegers)
    model.xtot = pyo.Var(range(P), domain=pyo.NonNegativeIntegers)
    model.y = pyo.Var(range(N), domain=pyo.Binary)

    def rule_pattern(model, n):
        return sum(l[p] * model.x[n, p] for p in range(P)) <= L * model.y[n]

    model.pattern = pyo.Constraint(range(N), rule=rule_pattern)

    # def rule_symmetry(model, n):
    #     return model.y[n] >= model.y[n + 1]

    # model.symmetry = pyo.Constraint(range(N - 1), rule=rule_symmetry)

    def rule_total(model, p):
        return sum(model.x[n, p] for n in range(N)) == model.xtot[p]

    model.total = pyo.Constraint(range(P), rule=rule_total)

    model.objexpr = sum(c * model.y[n] for n in range(N))

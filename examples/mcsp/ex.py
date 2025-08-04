import pyomo.environ as pyo

from mcsp import master_problem, sub_problem
from params import McspParams, create_single, create_random


def main(param: McspParams):
    model = extended_form(param)
    solver = pyo.SolverFactory("appsi_highs")
    solver.solve(model, tee=True)


def extended_form(param: McspParams):
    model = pyo.ConcreteModel()

    K = param.K
    P = param.P
    N = param.N
    d = param.d
    L = param.L
    c = param.c
    l = param.l
    master_problem(model, N, P, d)

    def rule_block(block, k):
        sub_problem(block, N[k], P, L[k], c[k], l)

    model.block = pyo.Block(range(K), rule=rule_block)

    def rule_x_equality(model, k, p):
        return model.xtot[k, p] == model.block[k].xtot[p]

    model.x_equality = pyo.Constraint(range(K), range(P), rule=rule_x_equality)

    model.obj = pyo.Objective(
        expr=sum(model.block[k].objexpr for k in range(K)), sense=pyo.minimize
    )

    return model


if __name__ == "__main__":
    K = 5
    P = 8
    param = create_random(K, P)
    main(param)

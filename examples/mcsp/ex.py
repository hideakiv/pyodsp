import pyomo.environ as pyo

from mcsp import master_problem, sub_problem


def main(
    N: list[int],
    P: int,
    d: list[int],
    L: list[int],
    c: list[float],
    l: list[int],
):
    model = extended_form(N, P, d, L, c, l)
    solver = pyo.SolverFactory("appsi_highs")
    solver.solve(model, tee=True)


def extended_form(
    N: list[int],
    P: int,
    d: list[int],
    L: list[int],
    c: list[float],
    l: list[int],
):
    model = pyo.ConcreteModel()

    K = len(N)
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
    N = [2000]
    P = 5
    d = [205, 2321, 143, 1089, 117]
    L = [110]
    c = [1.0]
    l = [70, 40, 55, 25, 35]
    main(N, P, d, L, c, l)

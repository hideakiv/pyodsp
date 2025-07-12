import pyomo.environ as pyo

from uc import balance, single_generator


def main(num_time: int, num_gens: int, demand: list[float], params: dict[int, dict]):
    model = extended_form(num_time, num_gens, demand, params)
    solver = pyo.SolverFactory("appsi_highs")
    solver.solve(model, tee=True)


def extended_form(
    num_time: int, num_gens: int, demand: list[float], params: dict[int, dict]
):
    model = pyo.ConcreteModel()

    balance(model, num_time, num_gens, demand)

    def rule_block(block, k):
        single_generator(block, num_time, params[k])

    model.block = pyo.Block(model.K, rule=rule_block)

    def rule_p_equality(model, k, t):
        return model.p[k, t] == model.block[k].p[t]

    model.p_equality = pyo.Constraint(model.K, model.T, rule=rule_p_equality)

    model.obj = pyo.Objective(
        expr=sum(model.block[k].objexpr for k in range(1, num_gens + 1)),
        sense=pyo.minimize,
    )

    return model


if __name__ == "__main__":
    pass

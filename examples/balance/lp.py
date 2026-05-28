from pathlib import Path
import json
import numpy as np
import pyomo.environ as pyo
from pyomo.core.base.block import generate_cuid_names

from balance import first_stage, mid_stage, last_stage
from scenarios import BalanceParams, create_static_scenarios


def solve(params: BalanceParams, solver="appsi_highs"):

    model = pyo.ConcreteModel()
    create_root(model, params)

    opt = pyo.SolverFactory(solver)
    opt.solve(model)

    # 1. Generate unique labels for your components
    labels = generate_cuid_names(model)

    # 2. Extract variable data into a dictionary
    solution_data = {}
    for var in model.component_data_objects(pyo.Var):
        solution_data[labels[var]] = var.value

    # 3. Save to disk
    with open("output/balance/lp/pyomo_solution.json", "w") as f:
        json.dump(solution_data, f)


def create_root(model: pyo.ConcreteModel, params: BalanceParams):
    first_stage(model, params.init_level, params.cp)
    model.next_stage = pyo.Block(
        range(params.cp.num_scenarios), rule=create_inner_rule(1, params)
    )

    def level_coupling_rule(m, s):
        return m.level[params.cp.time - 1] == m.next_stage[s].current_level

    model.level_coupling_cn = pyo.Constraint(
        range(params.cp.num_scenarios), rule=level_coupling_rule
    )

    def power_coupling_rule(m, s, t):
        return m.power[t] == m.next_stage[s].planned_power

    model.power_coupling_cn = pyo.Constraint(
        range(params.cp.num_scenarios), params.cp.time, rule=power_coupling_rule
    )

    # define cost
    objexpr = sum(
        params.cp.proc_prices[t] * model.procurement[t] for t in range(params.cp.time)
    )
    for s in range(params.cp.num_scenarios):
        objexpr += params.tp_dict[0][0, s] * model.next_stage[s].obj

    model.obj = pyo.Objective(expr=objexpr, sense=pyo.minimize)


def create_inner_rule(stage: int, params: BalanceParams):
    def create_inner(block: pyo.Block, s1: int):
        sp = params.sp_dict[stage][s1]
        block.current_level = pyo.Var()
        block.planned_power = pyo.Var(range(params.cp.time))
        mid_stage(block, block.current_level, block.power, sp, params.cp)
        if stage < params.cp.num_stages - 1:
            block.next_stage = pyo.Block(
                range(params.cp.num_scenarios),
                rule=create_inner_rule(stage + 1, params),
            )
        else:
            block.next_stage = pyo.Block(
                range(params.cp.num_scenarios),
                rule=create_leaf_rule(stage + 1, params),
            )

        def level_coupling_rule(m, s):
            return m.level[params.cp.time - 1] == m.next_stage[s].current_level

        block.level_coupling_cn = pyo.Constraint(
            range(params.cp.num_scenarios), rule=level_coupling_rule
        )

        def power_coupling_rule(m, s, t):
            return m.power[t] == m.next_stage[s].planned_power

        block.power_coupling_cn = pyo.Constraint(
            range(params.cp.num_scenarios), params.cp.time, rule=power_coupling_rule
        )

        objexpr = block.obj_expr
        for s in range(params.cp.num_scenarios):
            objexpr += params.tp_dict[stage][s1, s] * block.next_stage[s].obj
        block.obj = objexpr

    return create_inner


def create_leaf_rule(stage: int, params: BalanceParams):
    def create_leaf(block: pyo.Block, s1: int):
        sp = params.sp_dict[stage][s1]
        block.current_level = pyo.Var()
        block.planned_power = pyo.Var(range(params.cp.time))
        last_stage(
            block,
            block.current_level,
            params.final_level,
            params.final_penalty,
            block.planned_power,
            sp,
            params.cp,
        )
        block.obj = block.obj_expr

    return create_leaf


if __name__ == "__main__":
    from regime import (
        NormalRegime,
        HotSunnyRegime,
        HotCloudyRegime,
        ColdSunnyRegime,
        ColdCloudyRegime,
        RegimeParams,
    )

    time = 48
    r1 = NormalRegime(time)
    r2 = HotSunnyRegime(time)
    regimes = [r1, r2]
    tm = np.asarray([[0.8, 0.2], [0.6, 0.4]])
    regime_params = RegimeParams(regimes, tm)
    params = create_static_scenarios(3, [1, 1], regime_params, 0)
    solve(params)

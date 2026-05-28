from dataclasses import dataclass
import pyomo.environ as pyo


@dataclass
class ConstantParams:
    num_stages: int
    num_scenarios: int
    time: int

    min_level: float
    max_level: float
    max_discharge: float
    max_charge: float
    charge_efficiency: float

    max_purchase: list[float]
    max_sell: list[float]

    proc_prices: list[float]
    max_proc: list[float]

    penalty: float


@dataclass
class ScenarioParams:
    demands: list[float]
    prices: list[float]


def stage(
    block: pyo.ScalarBlock,
    current_level: pyo.Var,
    cp: ConstantParams,
):

    block.nextT = pyo.RangeSet(0, cp.time - 1)

    # define hydro

    def discharge_bd(block, t):
        return (0, cp.max_discharge)

    block.power_discharge = pyo.Var(
        block.nextT, domain=pyo.NonNegativeReals, bounds=discharge_bd
    )

    def charge_bd(block, t):
        return (0, cp.max_charge)

    block.power_charge = pyo.Var(
        block.nextT, domain=pyo.NonNegativeReals, bounds=charge_bd
    )

    # define level
    def level_bd(block, t):
        return (cp.min_level, cp.max_level)

    block.level = pyo.Var(block.nextT, domain=pyo.NonNegativeReals, bounds=level_bd)

    def level_rule(block, t):
        if t == 0:
            return (
                block.level[t]
                == current_level
                - block.power_discharge[t]
                + cp.charge_efficiency * block.power_charge[t]
            )
        else:
            return (
                block.level[t]
                == block.level[t - 1]
                - block.power_discharge[t]
                + cp.charge_efficiency * block.power_charge[t]
            )

    block.level_cn = pyo.Constraint(block.nextT, rule=level_rule)

    # define procurement
    def proc_bd(block, t):
        return (0, cp.max_proc[t])

    block.procurement = pyo.Var(
        block.nextT, domain=pyo.NonNegativeReals, bounds=proc_bd
    )

    # define power
    block.power = pyo.Var(block.nextT, domain=pyo.Reals)

    def power_rule(block, t):
        return (
            block.power[t]
            == block.procurement[t] + block.power_discharge[t] - block.power_charge[t]
        )

    block.power_cn = pyo.Constraint(block.nextT, rule=power_rule)


def first_stage(
    block: pyo.ScalarBlock,
    current_level: float,
    cp: ConstantParams,
):

    block.current_level = pyo.Var(domain=pyo.NonNegativeReals)
    block.current_level.fix(current_level)
    stage(block, block.current_level, cp)

    # define cost
    objexpr = sum(cp.proc_prices[t] * block.procurement[t] for t in range(cp.time))

    block.obj = pyo.Objective(expr=objexpr, sense=pyo.minimize)


def mid_stage(
    block: pyo.ScalarBlock,
    current_level: pyo.Var,
    planned_power: pyo.Var,
    sp: ScenarioParams,
    cp: ConstantParams,
):
    block.T = pyo.RangeSet(0, cp.time - 1)

    # define balance
    def market_bd(block, t):
        return (-cp.max_sell[t], cp.max_purchase[t])

    block.market = pyo.Var(block.T, domain=pyo.Reals, bounds=market_bd)

    block.violation_p = pyo.Var(block.T, domain=pyo.NonNegativeReals)
    block.violation_m = pyo.Var(block.T, domain=pyo.NonNegativeReals)

    def balance_rule(block, t):
        return (
            block.market[t]
            + planned_power[t]
            + block.violation_p[t]
            - block.violation_m[t]
            == sp.demands[t]
        )

    block.balance_cn = pyo.Constraint(block.T, rule=balance_rule)

    stage(block, current_level, cp)

    # define cost
    proc_cost = sum(cp.proc_prices[t] * block.procurement[t] for t in range(cp.time))
    market_cost = sum(sp.prices[t] * block.market[t] for t in range(cp.time))
    violation_cost = cp.penalty * sum(
        block.violation_p[t] + block.violation_m[t] for t in range(cp.time)
    )
    block.obj_expr = proc_cost + market_cost + violation_cost


def last_stage(
    block: pyo.ScalarBlock,
    current_level: pyo.Var,
    final_level: float,
    level_penalty: float,
    planned_power: pyo.Var,
    sp: ScenarioParams,
    cp: ConstantParams,
):
    block.T = pyo.RangeSet(0, cp.time - 1)

    # define balance
    def market_bd(block, t):
        return (-cp.max_sell[t], cp.max_purchase[t])

    block.market = pyo.Var(block.T, domain=pyo.Reals, bounds=market_bd)

    block.violation_p = pyo.Var(block.T, domain=pyo.NonNegativeReals)
    block.violation_m = pyo.Var(block.T, domain=pyo.NonNegativeReals)

    def balance_rule(block, t):
        return (
            block.market[t]
            + planned_power[t]
            + block.violation_p[t]
            - block.violation_m[t]
            == sp.demands[t]
        )

    block.balance_cn = pyo.Constraint(block.T, rule=balance_rule)

    # define level_violation
    block.level_violation_p = pyo.Var(domain=pyo.NonNegativeReals)
    block.level_violation_m = pyo.Var(domain=pyo.NonNegativeReals)

    def final_level_rule(block):
        return (
            current_level + block.level_violation_p - block.level_violation_m
            == final_level
        )

    block.final_level_cn = pyo.Constraint(rule=final_level_rule)

    # define cost
    market_cost = sum(sp.prices[t] * block.market[t] for t in range(cp.time))
    violation_cost = cp.penalty * sum(
        block.violation_p[t] + block.violation_m[t] for t in range(cp.time)
    )
    level_violation_cost = level_penalty * (
        block.level_violation_p + block.level_violation_m
    )
    block.obj_expr = market_cost + violation_cost + level_violation_cost

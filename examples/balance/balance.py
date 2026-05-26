from dataclasses import dataclass
import pyomo.environ as pyo


@dataclass
class Spec:
    min_level: float
    max_level: float
    max_discharge: float
    max_charge: float
    charge_efficiency: float

    penalty: float


@dataclass
class Schedule:
    time: int
    demands: list[float]
    prices: list[float]
    max_purchase: list[float]
    max_sell: list[float]

    next_time: int
    proc_prices: list[float]
    max_proc: list[float]


def stage(
    block: pyo.ScalarBlock,
    current_level: pyo.Var,
    schedule: Schedule,
    spec: Spec,
):

    block.nextT = pyo.RangeSet(0, schedule.next_time - 1)

    # define hydro

    def discharge_bd(block, t):
        return (0, spec.max_discharge)

    block.power_discharge = pyo.Var(
        block.nextT, domain=pyo.NonNegativeReals, bounds=discharge_bd
    )

    def charge_bd(block, t):
        return (0, spec.max_charge)

    block.power_charge = pyo.Var(
        block.nextT, domain=pyo.NonNegativeReals, bounds=charge_bd
    )

    # define level
    def level_bd(block, t):
        return (spec.min_level, spec.max_level)

    block.level = pyo.Var(block.nextT, domain=pyo.NonNegativeReals, bounds=level_bd)

    def level_rule(block, t):
        if t == 0:
            return (
                block.level[t]
                == current_level
                - block.power_discharge[t]
                + spec.charge_efficiency * block.power_charge[t]
            )
        else:
            return (
                block.level[t]
                == block.level[t - 1]
                - block.power_discharge[t]
                + spec.charge_efficiency * block.power_charge[t]
            )

    block.level_cn = pyo.Constraint(block.nextT, rule=level_rule)

    # define procurement
    def proc_bd(block, t):
        return (0, schedule.max_proc[t])

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


def root_stage(
    block: pyo.ScalarBlock,
    current_level: float,
    schedule: Schedule,
    spec: Spec,
):

    block.current_level = pyo.Var(domain=pyo.NonNegativeReals)
    block.current_level.fix(current_level)
    stage(block, block.current_level, schedule, spec)

    # define cost
    objexpr = sum(
        schedule.proc_prices[t] * block.procurement[t]
        for t in range(schedule.next_time)
    )

    block.obj = pyo.Objective(expr=objexpr, sense=pyo.minimize)


def nonroot_stage(
    block: pyo.ScalarBlock,
    current_level: pyo.Var,
    power: pyo.Var,
    schedule: Schedule,
    spec: Spec,
):
    block.T = pyo.RangeSet(0, schedule.time - 1)

    # define balance
    def market_bd(block, t):
        return (-schedule.max_sell[t], schedule.max_purchase[t])

    block.market = pyo.Var(block.T, domain=pyo.Reals, bounds=market_bd)

    block.violation_p = pyo.Var(block.T, domain=pyo.NonNegativeReals)
    block.violation_m = pyo.Var(block.T, domain=pyo.NonNegativeReals)

    def balance_rule(block, t):
        return (
            block.market[t] + power[t] + block.violation_p[t] - block.violation_m[t]
            == schedule.demands[t]
        )

    block.balance_cn = pyo.Constraint(block.T, rule=balance_rule)

    stage(block, current_level, schedule, spec)

    # define cost
    proc_cost = sum(
        schedule.proc_prices[t] * block.next_procurement[t]
        for t in range(schedule.next_time)
    )
    purchase_cost = sum(
        schedule.prices[t] * block.market[t] for t in range(schedule.time)
    )
    violation_cost = spec.penalty * sum(
        block.violation_p[t] + block.violation_m[t] for t in range(schedule.time)
    )
    block.obj_expr = proc_cost + purchase_cost + violation_cost

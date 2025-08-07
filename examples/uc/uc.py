import pyomo.environ as pyo
from params import UcParams


def balance(
    model: pyo.ConcreteModel, num_time: int, num_gens: int, demand: list[float]
):
    model.T = pyo.RangeSet(num_time)
    model.K = pyo.RangeSet(num_gens)

    model.p = pyo.Var(model.K, model.T, domain=pyo.NonNegativeReals)

    def rule_balance(model: pyo.ConcreteModel, t: int):
        return sum(model.p[k, t] for k in model.K) == demand[t - 1]

    model.balance = pyo.Constraint(model.T, rule=rule_balance)


def single_generator(block: pyo.Block, num_time: int, params: UcParams):
    num_seg = len(params.lp)
    neg = -max(params.DT_cold, params.UT)
    # Sets
    block.Tneg = pyo.RangeSet(neg, num_time)
    block.T = pyo.RangeSet(num_time)
    block.T0 = pyo.RangeSet(0, num_time)
    block.L = pyo.RangeSet(num_seg)

    # Variables
    block.u = pyo.Var(block.Tneg, domain=pyo.Binary)
    block.v = pyo.Var(block.Tneg, domain=pyo.Binary)
    block.w = pyo.Var(block.Tneg, domain=pyo.Binary)
    block.p = pyo.Var(block.T0, domain=pyo.NonNegativeReals)
    block.pl = pyo.Var(block.L, block.T, domain=pyo.NonNegativeReals)
    block.delta_hot = pyo.Var(block.T, domain=pyo.Binary)
    block.delta_cold = pyo.Var(block.T, domain=pyo.Binary)

    # logical
    def rule_garver_3bin(b, t):
        return b.u[t] - b.u[t - 1] == b.v[t] - b.w[t]

    block.garver_3bin = pyo.Constraint(block.T, rule=rule_garver_3bin)

    # initial
    u0 = params.u0
    p0 = params.p0

    block.u0 = pyo.Constraint(expr=block.u[0] == u0)
    block.p0 = pyo.Constraint(expr=block.p[0] == p0)

    def rule_init_v(b, t):
        if u0 == 1 and t == params.last_dn + 1:
            return b.v[t] == 1
        else:
            return b.v[t] == 0

    def rule_init_w(b, t):
        if u0 == 0 and t == params.last_up + 1:
            return b.w[t] == 1
        else:
            return b.w[t] == 0

    block.init_v = pyo.Constraint(range(neg, 0), rule=rule_init_v)
    block.init_w = pyo.Constraint(range(neg, 0), rule=rule_init_w)

    # min up/dn
    UT = params.UT
    DT = params.DT

    def rule_min_up(b, t):
        return sum(b.v[tt] for tt in range(t - UT + 1, t + 1)) <= b.u[t]

    def rule_min_dn(b, t):
        return sum(b.w[tt] for tt in range(t - DT + 1, t + 1)) <= 1 - b.u[t]

    block.min_up = pyo.Constraint(block.T, rule=rule_min_up)
    block.min_dn = pyo.Constraint(block.T, rule=rule_min_dn)

    # generation limits
    P_up = params.P_up
    P_dn = params.P_dn

    def rule_limit_up(b, t):
        return b.p[t] <= P_up * b.u[t]

    def rule_limit_dn(b, t):
        return b.p[t] >= P_dn * b.u[t]

    block.limit_up = pyo.Constraint(block.T, rule=rule_limit_up)
    block.limit_dn = pyo.Constraint(block.T, rule=rule_limit_dn)

    # ramp up/dn
    RU = params.RU
    SU = params.SU
    RD = params.RD
    SD = params.SD

    def rule_ramp_up(b, t):
        return b.p[t] - b.p[t - 1] <= RU * b.u[t - 1] + SU * b.v[t]

    def rule_ramp_dn(b, t):
        return b.p[t - 1] - b.p[t] <= RD * b.u[t] + SD * b.w[t]

    block.ramp_up = pyo.Constraint(block.T, rule=rule_ramp_up)
    block.ramp_dn = pyo.Constraint(block.T, rule=rule_ramp_dn)

    # power cost
    cp = params.cp
    lp = params.lp

    def rule_segments(b, l, t):
        if l == 1:
            return b.pl[l, t] <= lp[l - 1]
        else:
            return b.pl[l, t] <= lp[l - 1] - lp[l - 2]

    def rule_p_tot(b, t):
        return b.p[t] == sum(b.pl[l, t] for l in b.L)

    block.segments = pyo.Constraint(block.L, block.T, rule=rule_segments)
    block.p_tot = pyo.Constraint(block.T, rule=rule_p_tot)

    p_cost = sum(
        sum(cp[l - 1] * block.pl[l, t] for l in range(1, num_seg + 1))
        for t in range(1, num_time + 1)
    )

    # startup cost
    c_hot = params.c_hot
    c_cold = params.c_cold
    DT_cold = params.DT_cold

    def rule_hot_start(b, t):
        return b.delta_hot[t] <= sum(b.w[t - i] for i in range(DT, DT_cold))

    def rule_mode(b, t):
        return b.delta_hot[t] + b.delta_cold[t] == b.v[t]

    block.hot_start = pyo.Constraint(block.T, rule=rule_hot_start)
    block.mode = pyo.Constraint(block.T, rule=rule_mode)

    start_cost = sum(
        c_hot * block.delta_hot[t] + c_cold * block.delta_cold[t]
        for t in range(1, num_time + 1)
    )

    # running cost
    c_run = params.c_run
    running_cost = sum(c_run * block.u[t] for t in range(1, num_time + 1))

    # shutdown cost
    c_shut = params.c_shut
    shutdown_cost = sum(c_shut * block.w[t] for t in range(1, num_time + 1))

    # total
    block.objexpr = p_cost + start_cost + running_cost + shutdown_cost

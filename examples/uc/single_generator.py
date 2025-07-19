from dataclasses import dataclass
from typing import Self
from params import UcParams, create_random
import pyomo.environ as pyo

from pyodsp.alg.cuts import OptimalityCut
from pyodsp.dec.dd.alg_leaf import DdAlgLeaf
from pyodsp.dec.dd.message import (
    DdInitDnMessage,
    DdInitUpMessage,
    DdFinalDnMessage,
    DdFinalUpMessage,
    DdDnMessage,
    DdUpMessage,
)


class UcAlgLeaf(DdAlgLeaf):
    def __init__(self, params: UcParams, num_time: int) -> None:
        self.solver = SingleGeneratorDp(params, num_time)

    def build(self) -> None:
        pass

    def pass_init_dn_message(self, message: DdInitDnMessage) -> None:
        if not message.get_is_minimize():
            raise ValueError("Inconsistent optimization sense")

    def get_init_up_message(self) -> DdInitUpMessage:
        return DdInitUpMessage()

    def pass_dn_message(self, message: DdDnMessage) -> None:
        solution = message.get_solution()
        self._update_objective(solution)

    def get_up_message(self) -> DdUpMessage:
        pass

    def pass_final_dn_message(self, message: DdFinalDnMessage) -> None:
        pass

    def get_final_up_message(self) -> DdFinalUpMessage:
        pass


@dataclass
class DpNode:
    h: int
    k: int
    start_mode: int
    shut_mode: int
    dist: float | None = None
    predecessor: Self | None = None
    ps: dict[int, float] | None = None


NodeLabel = tuple[int, int, int, int]


class SingleGeneratorDp:
    def __init__(self, params: UcParams, num_time: int) -> None:
        self.params = params
        self.num_time = num_time
        self.mod = {t: 0.0 for t in range(1, num_time + 1)}
        self.model = self._create_model()

    def _create_model(self) -> pyo.AbstractModel:
        m = pyo.AbstractModel()
        m.h = pyo.Param(within=pyo.NonNegativeIntegers)
        m.h1 = pyo.Param(within=pyo.NonNegativeIntegers)
        m.k = pyo.Param(within=pyo.NonNegativeIntegers)
        m.seg = pyo.Param(within=pyo.NonNegativeIntegers)
        m.start_mode = pyo.Param(within=pyo.NonNegativeIntegers)
        m.shut_mode = pyo.Param(within=pyo.NonNegativeIntegers)

        m.T = pyo.RangeSet(m.h, m.k)
        m.T1 = pyo.RangeSet(m.h1, m.k)
        m.L = pyo.RangeSet(1, m.seg)

        m.p = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.pl = pyo.Var(m.L, m.T, domain=pyo.NonNegativeReals)

        # generation limits
        m.P_up = pyo.Param(m.T)
        m.P_dn = pyo.Param(m.T)

        def rule_limit_up(b, t):
            return b.p[t] <= b.P_up[t]

        def rule_limit_dn(b, t):
            return b.p[t] >= b.P_dn[t]

        m.limit_up = pyo.Constraint(m.T, rule=rule_limit_up)
        m.limit_dn = pyo.Constraint(m.T, rule=rule_limit_dn)

        # ramp up/dn
        m.RU = pyo.Param(within=pyo.NonNegativeReals)
        m.SU = pyo.Param(within=pyo.NonNegativeReals)
        m.RD = pyo.Param(within=pyo.NonNegativeReals)
        m.SD = pyo.Param(within=pyo.NonNegativeReals)

        def rule_ramp_up(b, t):
            return b.p[t] - b.p[t - 1] <= b.RU

        def rule_ramp_dn(b, t):
            return b.p[t - 1] - b.p[t] <= b.RD

        def rule_start_up(b):
            if b.start_mode > 0:
                return b.p[b.h] <= b.SU
            else:
                return pyo.Constraint.Skip

        def rule_shut_dn(b):
            if b.shut_mode > 0:
                return b.p[b.k] <= b.SD
            else:
                return pyo.Constraint.Skip

        m.ramp_up = pyo.Constraint(m.T1, rule=rule_ramp_up)
        m.ramp_dn = pyo.Constraint(m.T1, rule=rule_ramp_dn)
        m.start_up = pyo.Constraint(rule=rule_start_up)
        m.shut_dn = pyo.Constraint(rule=rule_shut_dn)

        # power cost
        m.cp = pyo.Param(m.L)
        m.lp = pyo.Param(m.L)

        def rule_segments(b, l, t):
            if l == 1:
                return b.pl[l, t] <= b.lp[l]
            else:
                return b.pl[l, t] <= b.lp[l] - b.lp[l - 1]

        def rule_p_tot(b, t):
            return b.p[t] == sum(b.pl[l, t] for l in b.L)

        m.segments = pyo.Constraint(m.L, m.T, rule=rule_segments)
        m.p_tot = pyo.Constraint(m.T, rule=rule_p_tot)

        m.c = pyo.Param(m.T)

        # total
        def rule_obj(b):
            return sum(
                b.c[t] * b.p[t] + sum(b.cp[l] * b.pl[l, t] for l in b.L) for t in b.T
            )

        m.obj = pyo.Objective(rule=rule_obj, sense=pyo.minimize)

        return m

    def solve_node(self, node: DpNode) -> float:
        h = node.h
        k = node.k
        start_mode = node.start_mode
        shut_mode = node.shut_mode
        # Convert cp and lp from list to dict for Pyomo
        cp_dict = {i + 1: v for i, v in enumerate(self.params.cp)}
        lp_dict = {i + 1: v for i, v in enumerate(self.params.lp)}
        data = {
            None: {
                "h": {None: h},
                "h1": {None: h + 1},
                "k": {None: k},
                "seg": {None: len(self.params.lp)},
                "start_mode": {None: start_mode},
                "shut_mode": {None: shut_mode},
                "RU": {None: self.params.RU},
                "SU": {None: self.params.SU},
                "RD": {None: self.params.RD},
                "SD": {None: self.params.SD},
                "cp": cp_dict,
                "lp": lp_dict,
            }
        }
        # Add time-dependent params (assume scalar for all t)
        data[None]["P_up"] = {t: self.params.P_up for t in range(h, k + 1)}
        data[None]["P_dn"] = {t: self.params.P_dn for t in range(h, k + 1)}
        data[None]["c"] = {t: self.mod[t] for t in range(h, k + 1)}

        # Create a ConcreteModel instance from the AbstractModel
        instance = self.model.create_instance(data)

        # Solve the model (assume a solver is available)
        solver = pyo.SolverFactory("appsi_highs")
        solver.solve(instance, tee=False)

        # Return the objective value (guaranteed float)
        obj = float(pyo.value(instance.obj))

        if start_mode == 1:
            obj += self.params.c_cold
        elif start_mode == 2:
            obj += self.params.c_hot

        obj += self.params.c_run * (k - h + 1)

        if shut_mode == 1:
            obj += self.params.c_shut

        ps = {}
        for t in range(h, k + 1):
            ps[t] = pyo.value(instance.p[t])
        node.dist += obj
        node.ps = ps

    def update_mod(self, new_mod: dict[int, float]) -> None:
        self.mod = new_mod

    def run(self) -> dict[int, float]:
        # k -> nodes
        reached_nodes: dict[int, dict[NodeLabel, DpNode]] = {}
        for k in range(self.num_time + 1):
            reached_nodes[k] = {}
        reached_nodes[0][0, 0, 0, 0] = DpNode(0, 0, 0, 0)
        final_val = None
        final_pred = None
        for k in range(self.num_time + 1):
            if k == 0:
                min_val = 0
                min_node = reached_nodes[0][0, 0, 0, 0]
            else:
                min_val = None
                min_node = None
                for node in reached_nodes[k].values():
                    self.solve_node(node)
                    if min_val is None or min_val > node.dist:
                        min_val = node.dist
                        min_node = node
            if min_node is None:
                continue

            # r, q, start_mode, shut_mode
            adjacent: list[DpNode] = self._get_adjacent(k)
            for node in adjacent:
                key = (node.h, node.k, node.start_mode, node.shut_mode)
                q = node.k
                if key in reached_nodes[q].keys():
                    next = reached_nodes[q][key]
                    if min_val < next.dist:
                        next.dist = min_val
                        next.predecessor = min_node
                else:
                    node.dist = min_val
                    node.predecessor = min_node
                    reached_nodes[q][key] = node
            if final_val is None or final_val > min_val:
                final_val = min_val
                final_pred = min_node
        pred = final_pred
        ps: dict[int, float] = {}
        while pred.predecessor is not None:
            ps.update(pred.ps)
            pred = pred.predecessor

        return ps

    def _get_adjacent(self, k: int) -> list[DpNode]:
        adjacent: list[DpNode] = []
        if k == 0:
            if self.params.u0 == 0:
                cold_min = self.params.last_up + self.params.DT_cold + 1
                can_be_up = self.params.last_up + self.params.DT + 1
                for r in range(max(1, can_be_up), self.num_time + 1):
                    for q in range(r + self.params.UT - 1, self.num_time + 1):
                        if r >= cold_min and q == self.num_time:
                            adjacent.append(DpNode(r, q, 1, 0))
                            adjacent.append(DpNode(r, q, 1, 1))
                        elif r >= cold_min and q < self.num_time:
                            adjacent.append(DpNode(r, q, 1, 1))
                        elif r < cold_min and q == self.num_time:
                            adjacent.append(DpNode(r, q, 2, 0))
                            adjacent.append(DpNode(r, q, 2, 1))
                        elif r < cold_min and q < self.num_time:
                            adjacent.append(DpNode(r, q, 2, 1))
            elif self.params.u0 == 1:
                r = 1
                must_be_up = self.params.last_dn + self.params.UT
                for q in range(max(1, must_be_up), self.num_time + 1):
                    if q == self.num_time:
                        adjacent.append(DpNode(r, q, 0, 1))
                        adjacent.append(DpNode(r, q, 0, 0))
                    elif q < self.num_time:
                        adjacent.append(DpNode(r, q, 0, 1))
        else:
            cold_min = k + self.params.DT_cold + 1
            for r in range(k + self.params.DT + 1, self.num_time + 1):
                for q in range(r + self.params.UT - 1, self.num_time + 1):
                    if r >= k + self.params.DT_cold + 1 and q == self.num_time:
                        adjacent.append(DpNode(r, q, 1, 0))
                        adjacent.append(DpNode(r, q, 1, 1))
                    elif r >= cold_min and q < self.num_time:
                        adjacent.append(DpNode(r, q, 1, 1))
                    elif r < cold_min and q == self.num_time:
                        adjacent.append(DpNode(r, q, 2, 0))
                        adjacent.append(DpNode(r, q, 2, 1))
                    elif r < cold_min and q < self.num_time:
                        adjacent.append(DpNode(r, q, 2, 1))
        return adjacent


if __name__ == "__main__":
    num_day = 1
    num_gens = 2
    num_seg = 5
    num_time, demand, params = create_random(num_day, num_gens, num_seg)

    test = SingleGeneratorDp(params[1], num_time)
    print(test.run())

from typing import List, Dict

from pyomo.environ import (
    ConcreteModel,
    Var,
    ScalarVar,
    Constraint,
    Objective,
    RangeSet,
    NonNegativeReals,
    NonNegativeIntegers,
    minimize,
    maximize,
    value,
)
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.alg.bm.cuts import OptimalityCut
from pyodsp.alg.bm.cuts_manager import CutInfo
from pyodsp.dec.dd.message import DdFinalDnMessage
from pyodsp.dec.dd.mip_heuristic_root import IMipHeuristicRoot


import pyomo.environ as pyo


def dp_sub_problem(model: pyo.Block, N: int, P: int, L: int, c: float, l: list[int]):
    model.x = pyo.Var(range(P), domain=pyo.NonNegativeIntegers)
    model.y = pyo.Var(domain=pyo.Binary)
    model.xtot = pyo.Var(range(P), domain=pyo.NonNegativeIntegers)

    def rule_pattern(model):
        return sum(l[p] * model.x[p] for p in range(P)) <= L * model.y

    model.pattern = pyo.Constraint(rule=rule_pattern)

    def rule_total(model, p):
        return N * model.x[p] == model.xtot[p]

    model.total = pyo.Constraint(range(P), rule=rule_total)

    model.objexpr = c * N * model.y


class DpHeuristic(IMipHeuristicRoot):
    def __init__(self, solver_config: SolverConfig, N: list[int]):
        self.solver_config = solver_config
        self.N = N

    def _create_master(self, model: ConcreteModel, solver_config: SolverConfig):
        for obj in model.component_objects(Objective, active=True):
            raise ValueError("Objective should not be defined in coupling model")
        if self.is_minimize:
            model._dd_obj = Objective(expr=0.0, sense=minimize)
        else:
            model._dd_obj = Objective(expr=0.0, sense=maximize)
        return PyomoSolver(model, solver_config, [])

    def build(self, **kwargs) -> None:
        self.groups: List[List[int]] = kwargs["groups"]
        coupling_model: ConcreteModel = kwargs["coupling_model"]
        self.cuts: List[List[CutInfo]] = kwargs["cuts"]
        self.vars_dn: Dict[int, List[ScalarVar]] = kwargs["vars_dn"]
        self.is_minimize: bool = kwargs["is_minimize"]
        self.master = self._create_master(coupling_model, self.solver_config)
        for cutlist, group in zip(self.cuts, self.groups):
            assert len(group) == 1
            idx = group[0]
            N = self.N[idx - 1]
            vars = self.vars_dn[idx]

            num_cuts = len(cutlist)
            minkowski_vars = Var(RangeSet(0, num_cuts - 1), domain=NonNegativeReals)
            self.master.model.add_component(f"_vars_{idx}", minkowski_vars)

            def equality_rule(m, i):
                lhs = 0.0
                for j, cutinfo in enumerate(cutlist):
                    lhs += minkowski_vars[j] * cutinfo.cut.info["solution"][i]
                return lhs == vars[i]

            self.master.model.add_component(
                f"_equality_{idx}",
                Constraint(RangeSet(0, len(vars) - 1), rule=equality_rule),
            )

            self.master.model.add_component(
                f"_convexity_{idx}",
                Constraint(
                    expr=sum(
                        minkowski_vars[i]
                        for i in range(num_cuts)
                        if isinstance(cutlist[i].cut, OptimalityCut)
                    )
                    == 1
                ),
            )

            integer_vars = Var(RangeSet(0, num_cuts - 1), domain=NonNegativeIntegers)
            self.master.model.add_component(f"_integer_vars_{idx}", integer_vars)

            def integer_rule(m, j):
                return integer_vars[j] == N * minkowski_vars[j]

            self.master.model.add_component(
                f"_integer_{idx}",
                Constraint(RangeSet(0, num_cuts - 1), rule=integer_rule),
            )

            obj = 0.0
            for j, cutinfo in enumerate(cutlist):
                if self.is_minimize:
                    rhs = cutinfo.constraint.upper
                else:
                    rhs = cutinfo.constraint.lower
                assert rhs is not None

                obj += rhs * minkowski_vars[j]

            self.master.model._dd_obj.expr += obj

    def run(self) -> Dict[int, DdFinalDnMessage]:
        self.master.solve()

        for cutlist, group in zip(self.cuts, self.groups):
            assert len(group) == 1
            idx = group[0]
            N = self.N[idx - 1]

            num_cuts = len(cutlist)
            xs = self.master.model.component(f"_integer_vars_{idx}")
            for j in range(num_cuts):
                cutinfo = cutlist[j]
                x = int(value(xs[j]))
                pattern = [round(sol / N) for sol in cutinfo.cut.info["solution"]]
                print(
                    f"group {idx}: ",
                    f"{x}\t x pattern:",
                    pattern,
                )

        # pass nothing to children
        solutions = {}
        for group in self.groups:
            assert len(group) == 1
            idx = group[0]
            solutions[idx] = DdFinalDnMessage(None)

        return solutions

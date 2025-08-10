from typing import List, Dict

from uc import UcParams, single_generator
from pyomo.environ import (
    Block,
    ConcreteModel,
    Var,
    ScalarVar,
    Constraint,
    Objective,
    RangeSet,
    NonNegativeReals,
    minimize,
    maximize,
    value,
)
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.alg.bm.cuts_manager import CutInfo
from pyodsp.dec.dd.message import DdFinalDnMessage
from pyodsp.dec.dd.mip_heuristic_root import IMipHeuristicRoot


class UcHeuristicRoot(IMipHeuristicRoot):
    def __init__(
        self, solver_config: SolverConfig, params: dict[int, UcParams], num_time: int
    ):
        self.solver_config = solver_config
        self.params = params
        self.num_time = num_time

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
            vars = self.vars_dn[idx]
            param = self.params[idx]

            block = Block()
            self.master.model.add_component(f"_block_{idx}", block)
            single_generator(block, self.num_time, param)

            num_cuts = len(cutlist)
            minkowski_vars = Var(RangeSet(0, num_cuts - 1), domain=NonNegativeReals)
            self.master.model.add_component(f"_vars_{idx}", minkowski_vars)

            def equality_rule(m, i):
                return block.p[i + 1] == vars[i]

            self.master.model.add_component(
                f"_equality_{idx}",
                Constraint(RangeSet(0, len(vars) - 1), rule=equality_rule),
            )

            def u_rule(m, i):
                lhs = 0.0
                for j, cutinfo in enumerate(cutlist):
                    if cutinfo.cut.info["solution"][i] > 1e-9:
                        lhs += minkowski_vars[j]
                return lhs == block.u[i + 1]

            self.master.model.add_component(
                f"_u_equality_{idx}",
                Constraint(RangeSet(0, len(vars) - 1), rule=u_rule),
            )

            self.master.model._dd_obj.expr += block.objexpr

    def run(self) -> Dict[int, DdFinalDnMessage]:
        self.master.solve()

        if self.master.is_optimal():
            solutions = {}
            for group in self.groups:
                assert len(group) == 1
                idx = group[0]
                solution = [value(var) for var in self.vars_dn[idx]]
                solutions[idx] = DdFinalDnMessage(solution)

            return solutions
        else:
            # TODO: use logging
            print("WARNING: Unable to find heuristic solution.")
            solutions = {}
            for group in self.groups:
                assert len(group) == 1
                idx = group[0]
                solutions[idx] = DdFinalDnMessage(None)

            return solutions

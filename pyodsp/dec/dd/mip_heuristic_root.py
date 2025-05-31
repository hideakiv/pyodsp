from typing import List, Dict

from pyomo.environ import (
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
from pyodsp.alg.cuts import OptimalityCut
from pyodsp.alg.cuts_manager import CutInfo
from .message import DdFinalDnMessage


class MipHeuristicRoot:
    def __init__(
        self,
        groups: List[List[int]],
        coupling_model: ConcreteModel,
        solver_config: SolverConfig,
        cuts: List[List[CutInfo]],
        vars_dn: Dict[int, List[ScalarVar]],
        is_minimize: bool,
    ):
        self.cuts = cuts
        self.vars_dn = vars_dn
        self.is_minimize = is_minimize
        self.groups = groups
        self.master = self._create_master(coupling_model, solver_config)

    def _create_master(self, model: ConcreteModel, solver_config: SolverConfig):
        for obj in model.component_objects(Objective, active=True):
            raise ValueError("Objective should not be defined in coupling model")
        if self.is_minimize:
            model._dd_obj = Objective(expr=0.0, sense=minimize)
        else:
            model._dd_obj = Objective(expr=0.0, sense=maximize)
        return PyomoSolver(model, solver_config, [])

    def build(self) -> None:
        for cutlist, group in zip(self.cuts, self.groups):
            assert len(group) == 1
            idx = group[0]
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
        solutions = {}

        for group in self.groups:
            assert len(group) == 1
            idx = group[0]
            solution = [value(var) for var in self.vars_dn[idx]]
            solutions[idx] = DdFinalDnMessage(solution)

        return solutions

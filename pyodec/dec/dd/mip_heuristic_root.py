from typing import List, Dict

from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    Objective,
    RangeSet,
    NonNegativeReals,
    minimize,
    maximize,
    value,
)
from .alg_root import DdAlgRoot
from pyodec.solver.pyomo_solver import PyomoSolver
from pyodec.alg.cuts import OptimalityCut


class MipHeuristicRoot:
    def __init__(self, solver: str, groups: List[List[int]], alg: DdAlgRoot, **kwargs):
        self.alg = alg
        self.groups = groups
        self.master = self._create_master(self.alg.coupling_model, solver, **kwargs)

    def _create_master(self, model: ConcreteModel, solver: str, **kwargs):
        for obj in model.component_objects(Objective, active=True):
            raise ValueError("Objective should not be defined in coupling model")
        if self.alg.is_minimize:
            model._dd_obj = Objective(expr=0.0, sense=minimize)
        else:
            model._dd_obj = Objective(expr=0.0, sense=maximize)
        return PyomoSolver(model, solver, [], **kwargs)

    def build(self) -> None:
        cuts = self.alg.get_cuts()

        for cutlist, group in zip(cuts, self.groups):
            assert len(group) == 1
            idx = group[0]
            vars = self.alg.get_vars_dn()[idx]

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
                if self.alg.is_minimize:
                    rhs = cutinfo.constraint.upper
                else:
                    rhs = cutinfo.constraint.lower
                assert rhs is not None

                obj += rhs * minkowski_vars[j]

            self.master.model._dd_obj.expr += obj

    def run(self) -> Dict[int, List[float]]:
        self.master.solve()
        solutions = {}

        for group in self.groups:
            assert len(group) == 1
            idx = group[0]
            solution = [value(var) for var in self.alg.get_vars_dn()[idx]]
            solutions[idx] = solution

        return solutions

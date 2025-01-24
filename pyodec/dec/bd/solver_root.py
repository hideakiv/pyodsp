from typing import List, Dict

from pyomo.environ import ConcreteModel, Var, Constraint, Objective, Reals, RangeSet
from pyomo.core.base.var import VarData

from .solver import BdSolver
from .cuts import OptimalityCut, FeasibilityCut


class BdSolverRoot(BdSolver):

    def __init__(self, model: ConcreteModel, solver: str, **kwargs):
        super().__init__(model, solver, **kwargs)
        self.tolerance = 1e-6

        self.optimality_cuts: Dict[int, List[Constraint]] = {}
        self.feasibility_cuts: Dict[int, List[Constraint]] = {}

    def build(self, subobj_bounds: List[float]):
        self.num_cuts = len(subobj_bounds)
        self._update_objective(subobj_bounds)
        for i in range(self.num_cuts):
            self.optimality_cuts[i] = []
            self.feasibility_cuts[i] = []

    def set_tolerance(self, tolerance) -> None:
        self.tolerance = tolerance

    def _update_objective(self, subobj_bounds: List[float]):
        def theta_bounds(model, i):
            if self.get_objective_sense():
                # Minimization
                return (subobj_bounds[i], None)
            else:
                # Maximization
                return (None, subobj_bounds[i])

        self.model._theta = Var(
            RangeSet(0, self.num_cuts - 1), domain=Reals, bounds=theta_bounds
        )

        self.original_objective.deactivate()

        modified_expr = self.original_objective.expr + sum(
            self.model._theta[i] for i in range(self.num_cuts)
        )

        self.model._bd_obj = Objective(
            expr=modified_expr, sense=self.original_objective.sense
        )

    def add_optimality_cut(
        self, i: int, cut: OptimalityCut, vars: List[VarData]
    ) -> bool:

        theta_val = self.model._theta[i].value
        cut_num = len(self.optimality_cuts[i])

        if self.get_objective_sense():
            # Minimization
            if theta_val >= cut.objective_value - self.tolerance:
                # No need to add the cut
                return False
            self.model.add_component(
                f"_optimality_cut_{i}_{cut_num}",
                Constraint(
                    expr=sum(cut.coefficients[i] * vars[i] for i in range(len(vars)))
                    + self.model._theta[i]
                    >= cut.constant
                ),
            )
        else:
            # Maximization
            if theta_val <= cut.objective_value + self.tolerance:
                # No need to add the cut
                return False
            self.model.add_component(
                f"_optimality_cut_{i}_{cut_num}",
                Constraint(
                    expr=sum(cut.coefficients[i] * vars[i] for i in range(len(vars)))
                    + self.model._theta[i]
                    <= cut.constant
                ),
            )

        self.optimality_cuts[i].append(
            self.model.component(f"_optimality_cut_{i}_{cut_num}")
        )
        return True

    def add_feasibility_cut(
        self, i: int, cut: FeasibilityCut, vars: List[VarData]
    ) -> bool:

        cut_num = len(self.feasibility_cuts[i])

        if self.get_objective_sense():
            # Minimization
            self.model.add_component(
                f"_feasibility_cut_{i}_{cut_num}",
                Constraint(
                    expr=sum(cut.coefficients[i] * vars[i] for i in range(len(vars)))
                    >= cut.constant
                ),
            )
        else:
            # Maximization
            self.model.add_component(
                f"_feasibility_cut_{i}_{cut_num}",
                Constraint(
                    expr=sum(cut.coefficients[i] * vars[i] for i in range(len(vars)))
                    <= cut.constant
                ),
            )

        self.feasibility_cuts[i].append(
            self.model.component(f"_feasibility_cut_{i}_{cut_num}")
        )
        return True

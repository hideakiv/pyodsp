from typing import List, Tuple

from pyomo.environ import (
    ConcreteModel,
    Objective,
)
from pyomo.core.base.var import VarData

from .solver import DdSolver


class DdSolverLeaf(DdSolver):
    def __init__(self, model: ConcreteModel, solver: str, **kwargs):
        super().__init__(model, solver, **kwargs)

    def build(self) -> None:
        self.original_objective.deactivate()

    def add_to_objective(
        self, coupling_vars_up: List[VarData], coeffs: List[float]
    ) -> None:
        modified_expr = self.original_objective.expr + sum(
            coeff * coupling_var
            for coeff, coupling_var in zip(coeffs, coupling_vars_up)
        )

        self.model._mod_obj = Objective(
            expr=modified_expr, sense=self.original_objective.sense
        )

    def get_solution_or_ray(
        self, coupling_vars: List[VarData]
    ) -> Tuple[bool, List[float]]:
        if self.is_optimal():
            return True, self.get_solution(coupling_vars)
        elif self.is_unbounded():
            raise NotImplementedError()
        else:
            raise ValueError("Unknown solver status")

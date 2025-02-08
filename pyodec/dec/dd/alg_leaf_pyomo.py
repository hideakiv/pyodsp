from typing import List, Tuple

from pyomo.core.base.constraint import ConstraintData

from .alg_leaf import DdAlgLeaf
from ..utils import CouplingData, get_nonzero_coefficients_from_model
from pyodec.solver.pyomo_solver import PyomoSolver
from pyodec.solver.pyomo_utils import (
    update_linear_terms_in_objective,
    create_restricted_mode,
    activate_restricted_mode,
    deactivate_restricted_mode,
)


class DdAlgLeafPyomo(DdAlgLeaf):
    def __init__(self, solver: PyomoSolver):
        self.solver = solver

    def build(self) -> None:
        self.solver.original_objective.deactivate()

    def update_objective(self, coeffs: List[float]) -> None:
        update_linear_terms_in_objective(self.solver, coeffs, self.solver.vars)

    def get_solution_or_ray(self) -> Tuple[bool, List[float], float]:
        self.solver.solve()
        if self.solver.is_optimal():
            solution = self.solver.get_solution()
            obj = self.solver.get_objective_value()
            return True, solution, obj
        elif self.solver.is_unbounded():
            ray, obj = self._get_ray()
            return False, ray, obj
        else:
            raise ValueError("Unknown solver status")

    def is_minimize(self) -> bool:
        return self.solver.is_minimize()

    def get_len_vars(self) -> int:
        return len(self.solver.vars)

    def get_objective_value(self) -> float:
        return self.solver.get_objective_value()

    def _get_ray(self) -> Tuple[List[float], float]:
        if self.solver.model.component("_restricted_constr") is None:
            coupling_vars = self.solver.vars
            coupling_info: List[CouplingData] = get_nonzero_coefficients_from_model(
                self.solver.model, coupling_vars
            )
            related_constraints: List[ConstraintData] = [
                coupling_data.constraint for coupling_data in coupling_info
            ]
            self.restricted_info = create_restricted_mode(
                self.solver, related_constraints
            )

        activate_restricted_mode(self.solver, self.restricted_info)
        self.solver.solve()

        ray = self.solver.get_solution()
        obj = self.solver.get_objective_value()

        deactivate_restricted_mode(self.solver, self.restricted_info)

        return ray, obj

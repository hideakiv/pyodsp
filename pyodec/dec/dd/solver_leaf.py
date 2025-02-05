from typing import List, Tuple

from pyodec.solver.pyomo_solver import PyomoSolver
from pyodec.solver.pyomo_utils import update_linear_terms_in_objective


class DdAlgLeaf:
    def __init__(self, solver: PyomoSolver):
        self.solver = solver

    def build(self) -> None:
        self.solver.original_objective.deactivate()

    def update_objective(self, coeffs: List[float]) -> None:
        update_linear_terms_in_objective(self.solver, coeffs, self.solver.vars)

    def get_solution_or_ray(self) -> Tuple[bool, List[float]]:
        self.solver.solve()
        if self.solver.is_optimal():
            return True, self.solver.get_solution()
        elif self.solver.is_unbounded():
            raise NotImplementedError()
        else:
            raise ValueError("Unknown solver status")

    def is_minimize(self) -> bool:
        return self.solver.is_minimize()

    def get_len_vars(self) -> int:
        return len(self.solver.vars)

    def get_objective_value(self) -> float:
        return self.solver.get_objective_value()

from typing import List, Tuple
from pathlib import Path

from .alg_leaf import DdAlgLeaf
from pyodsp.solver.pyomo_solver import PyomoSolver
from pyodsp.solver.pyomo_utils import update_linear_terms_in_objective


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

        ray = self.solver.get_unbd_ray()
        obj = self.solver.get_unbounded_model_objective_value()

        return ray, obj

    def fix_variables_and_solve(self, values: List[float]) -> None:
        """Fix the variables to a specified value and then solve

        Args:
            values: The values to be set.
        """
        self.coupling_values: List[float] = values
        for i, var in enumerate(self.solver.vars):
            var.fix(values[i])
        self.solver.solve()

    def save(self, dir: Path) -> None:
        self.solver.save(dir)

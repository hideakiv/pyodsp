from typing import List, Dict, Tuple
from pathlib import Path
import time
import pandas as pd

from pyodsp.alg.cuts import Cut, OptimalityCut, FeasibilityCut
from pyodsp.alg.params import DEC_CUT_ABS_TOL

from .alg_leaf import DdAlgLeaf
from .coupling_manager import CouplingManager
from pyodsp.solver.pyomo_solver import PyomoSolver
from pyodsp.solver.pyomo_utils import update_linear_terms_in_objective


class DdAlgLeafPyomo(DdAlgLeaf):
    def __init__(self, solver: PyomoSolver):
        self.solver = solver
        self.step_time: List[float] = []
        self._is_minimize = self.solver.is_minimize()

    def set_coupling_matrix(self, coupling_matrix: List[Dict[int, float]]) -> None:
        self.cm = CouplingManager(coupling_matrix, self.get_len_vars(), self.is_minimize())

    def build(self) -> None:
        self.solver.original_objective.deactivate()

    def pass_solution(self, solution: List[float]) -> None:
        self._update_objective(solution)

    def _update_objective(self, coeffs: List[float]) -> None:
        self.primal_coeffs = self.cm.dual_times_matrix(coeffs)
        update_linear_terms_in_objective(self.solver, self.primal_coeffs, self.solver.vars)

    def get_subgradient(self) -> Cut:
        is_optimal, solution, obj = self.get_solution_or_ray()
        if is_optimal:
            dual_coeffs = self.cm.matrix_times_primal(solution)
            product = self.cm.inner_product(self.primal_coeffs, solution)
            rhs = obj - product
            sparse_coeff = {j: val for j, val in enumerate(dual_coeffs) if abs(val) > DEC_CUT_ABS_TOL}
            return OptimalityCut(
                coeffs=sparse_coeff,
                rhs=rhs,
                objective_value=obj,
                info={"solution": solution},
            )
        else:
            dual_coeffs = self.cm.matrix_times_primal(solution)
            product = self.cm.inner_product(self.primal_coeffs, solution)
            rhs = obj - product
            sparse_coeff = {j: val for j, val in enumerate(dual_coeffs) if abs(val) > DEC_CUT_ABS_TOL}
            return FeasibilityCut(
                coeffs=sparse_coeff, rhs=rhs, info={"solution": solution}
            )

    def get_solution_or_ray(self) -> Tuple[bool, List[float], float]:
        start = time.time()
        self.solver.solve()
        if self.solver.is_optimal():
            solution = self.solver.get_solution()
            obj = self.solver.get_objective_value()
            self.step_time.append(time.time() - start)
            return True, solution, obj
        elif self.solver.is_unbounded():
            ray, obj = self._get_ray()
            self.step_time.append(time.time() - start)
            return False, ray, obj
        else:
            raise ValueError("Unknown solver status")

    def is_minimize(self) -> bool:
        return self._is_minimize

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
        self.solver.activate_original_objective()
        self.solver.solve()

    def save(self, dir: Path) -> None:
        self.solver.save(dir)
        path = dir / "step_time.csv"
        df = pd.DataFrame(self.step_time, columns=["step_time"])
        df.to_csv(path, index=False)

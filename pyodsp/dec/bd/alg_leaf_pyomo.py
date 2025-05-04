from typing import List
from pathlib import Path
import time
import pandas as pd

from pyomo.environ import Suffix
from pyomo.core.base.constraint import ScalarConstraint

from .message import BdInitMessage, BdFinalMessage, BdDnMessage
from .alg_leaf import BdAlgLeaf
from ..utils import CouplingData, get_nonzero_coefficients_from_model
from pyodsp.alg.cuts import Cut, OptimalityCut, FeasibilityCut
from pyodsp.solver.pyomo_solver import PyomoSolver
from pyodsp.alg.params import DEC_CUT_ABS_TOL


class BdAlgLeafPyomo(BdAlgLeaf):
    def __init__(self, solver: PyomoSolver):
        self.solver = solver
        self.solver.model.dual = Suffix(direction=Suffix.IMPORT)
        self.step_time: List[float] = []

    def build(self) -> None:
        coupling_vars = self.solver.vars
        self.coupling_info: List[CouplingData] = get_nonzero_coefficients_from_model(
            self.solver.model, coupling_vars
        )
        self.coupling_constraints: List[ScalarConstraint] = [
            coupling_data.constraint for coupling_data in self.coupling_info
        ]

    def pass_init_message(self, message: BdInitMessage) -> None:
        pass

    def pass_dn_message(self, message: BdDnMessage) -> None:
        solution = message.get_solution()
        self._fix_variables(solution)

    def pass_final_message(self, message: BdFinalMessage) -> None:
        pass

    def _fix_variables(self, coupling_values: List[float]) -> None:
        """Fix the variables to a specified value

        Args:
            vars: The variables to be fixed.
            values: The values to be set.
        """
        self.coupling_values: List[float] = coupling_values
        for i, var in enumerate(self.solver.vars):
            var.fix(coupling_values[i])

    def get_subgradient(self) -> Cut:
        start = time.time()
        self.solver.solve()
        cut = self._get_subgradient_inner()
        self.step_time.append(time.time() - start)
        return cut

    def _get_subgradient_inner(self) -> Cut:
        if self.solver.is_optimal():
            cut = self._optimality_cut()
            return cut
        elif self.solver.is_infeasible():
            cut = self._feasibility_cut()
            return cut
        else:
            raise ValueError("Unknown solver status")

    def get_objective_value(self) -> float:
        return self.solver.get_objective_value()

    def _optimality_cut(self) -> OptimalityCut:
        pi = self.solver.get_dual(self.coupling_constraints)
        objective = self.get_objective_value()
        coeff = [0.0 for _ in range(len(self.solver.vars))]
        rhs = objective
        for i, dual_var in enumerate(pi):
            coupling_data = self.coupling_info[i]
            for j, coefficients in coupling_data.coefficients.items():
                temp = dual_var * coefficients
                coeff[j] += temp
                rhs += temp * self.coupling_values[j]
        sparse_coeff = {
            j: val for j, val in enumerate(coeff) if abs(val) > DEC_CUT_ABS_TOL
        }
        return OptimalityCut(
            coeffs=sparse_coeff, rhs=rhs, objective_value=objective, info={}
        )

    def _feasibility_cut(self) -> FeasibilityCut:
        sigma = self.solver.get_dual_ray(self.coupling_constraints)

        objective = self.solver.get_infeasible_model_objective_value()

        coeff = [0.0 for _ in range(len(self.solver.vars))]
        rhs = objective
        for i, dual_ray in enumerate(sigma):
            coupling_data = self.coupling_info[i]
            for j, coefficients in coupling_data.coefficients.items():
                temp = dual_ray * coefficients
                coeff[j] += temp
                rhs += temp * self.coupling_values[j]
        sparse_coeff = {
            j: val for j, val in enumerate(coeff) if abs(val) > DEC_CUT_ABS_TOL
        }
        return FeasibilityCut(coeffs=sparse_coeff, rhs=rhs, info={})

    def save(self, dir: Path) -> None:
        self.solver.save(dir)
        path = dir / "step_time.csv"
        df = pd.DataFrame(self.step_time, columns=["step_time"])
        df.to_csv(path, index=False)

    def is_minimize(self) -> bool:
        return self.solver.is_minimize()

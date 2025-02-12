from typing import List
from pathlib import Path

from pyomo.environ import Suffix, value
from pyomo.core.base.constraint import ConstraintData

from .alg_leaf import BdAlgLeaf
from ..utils import CouplingData, get_nonzero_coefficients_from_model
from pyodec.alg.cuts import Cut, OptimalityCut, FeasibilityCut
from pyodec.solver.pyomo_solver import PyomoSolver


class BdAlgLeafPyomo(BdAlgLeaf):
    def __init__(self, solver: PyomoSolver):
        self.solver = solver
        self.solver.model.dual = Suffix(direction=Suffix.IMPORT)

    def build(self) -> None:
        coupling_vars = self.solver.vars
        self.coupling_info: List[CouplingData] = get_nonzero_coefficients_from_model(
            self.solver.model, coupling_vars
        )
        self.coupling_constraints: List[ConstraintData] = [
            coupling_data.constraint for coupling_data in self.coupling_info
        ]

    def fix_variables(self, values: List[float]) -> None:
        """Fix the variables to a specified value

        Args:
            vars: The variables to be fixed.
            values: The values to be set.
        """
        self.coupling_values: List[float] = values
        for i, var in enumerate(self.solver.vars):
            var.fix(values[i])

    def get_subgradient(self) -> Cut:
        self.solver.solve()
        if self.solver.is_optimal():
            return self._optimality_cut()
        elif self.solver.is_infeasible():
            return self._feasibility_cut()
        else:
            raise ValueError("Unknown solver status")

    def _optimality_cut(self) -> OptimalityCut:
        pi = self.solver.get_dual(self.coupling_constraints)
        objective = self.solver.get_objective_value()
        coeff = [0.0 for _ in range(len(self.solver.vars))]
        rhs = objective
        for i, dual_var in enumerate(pi):
            coupling_data = self.coupling_info[i]
            for j, coefficients in coupling_data.coefficients.items():
                temp = dual_var * coefficients
                coeff[j] += temp
                rhs += temp * self.coupling_values[j]
        return OptimalityCut(coeffs=coeff, rhs=rhs, objective_value=objective, info={})

    def _feasibility_cut(self) -> FeasibilityCut:
        sigma = self.solver.get_dual_ray(self.coupling_constraints)

        objective = value(self.solver._infeasible_model._infeasible_obj)

        coeff = [0.0 for _ in range(len(self.solver.vars))]
        rhs = objective
        for i, dual_ray in enumerate(sigma):
            coupling_data = self.coupling_info[i]
            for j, coefficients in coupling_data.coefficients.items():
                temp = dual_ray * coefficients
                coeff[j] += temp
                rhs += temp * self.coupling_values[j]
        return FeasibilityCut(coeffs=coeff, rhs=rhs, info={})

    def save(self, dir: Path) -> None:
        self.solver.save(dir)

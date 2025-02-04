from typing import List

from pyomo.environ import Var, Objective

from pyodec.solver.pyomo_solver import PyomoSolver


def add_linear_terms_to_objective(
    solver: PyomoSolver, coeffs: List[float], vars: Var
) -> None:
    solver.original_objective.deactivate()
    modified_expr = solver.original_objective.expr + sum(
        coeffs[i] * vars[i] for i in range(len(coeffs))
    )

    solver.model._mod_obj = Objective(
        expr=modified_expr, sense=solver.original_objective.sense
    )


def add_terms_to_objective(solver: PyomoSolver, vars: Var) -> None:
    coeffs = [1.0 for _ in range(len(vars))]
    add_linear_terms_to_objective(solver, coeffs, vars)

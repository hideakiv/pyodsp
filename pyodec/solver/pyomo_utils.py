from typing import List

from pyomo.environ import Var, Objective
from pyomo.core.base.var import VarData

from pyodec.solver.pyomo_solver import PyomoSolver


def add_linear_terms_to_objective(
    solver: PyomoSolver, coeffs: List[float], vars: Var | List[VarData]
) -> None:
    solver.original_objective.deactivate()
    update_linear_terms_in_objective(solver, coeffs, vars)


def update_linear_terms_in_objective(
    solver: PyomoSolver, coeffs: List[float], vars: Var | List[VarData]
) -> None:
    modified_expr = solver.original_objective.expr + sum(
        coeffs[i] * vars[i] for i in range(len(coeffs))
    )

    if solver.model.component("_mod_obj") is not None:
        solver.model.del_component("_mod_obj")

    solver.model._mod_obj = Objective(
        expr=modified_expr, sense=solver.original_objective.sense
    )


def add_terms_to_objective(solver: PyomoSolver, vars: Var) -> None:
    coeffs = [1.0 for _ in range(len(vars))]
    add_linear_terms_to_objective(solver, coeffs, vars)


def update_quad_terms_in_objective(
    solver: PyomoSolver,
    quadvars: List[VarData],
    center: List[float],
    penalty: float = 1.0,
) -> None:
    if solver.original_objective.sense > 0:
        modified_expr = solver.model._mod_obj.expr + 0.5 * sum(
            penalty * (quadvar - val) ** 2 for quadvar, val in zip(quadvars, center)
        )
    else:
        modified_expr = solver.model._mod_obj.expr - 0.5 * sum(
            penalty * (quadvar - val) ** 2 for quadvar, val in zip(quadvars, center)
        )

    if solver.model.component("_mod_quad_obj") is not None:
        solver.model.del_component("_mod_quad_obj")

    solver.model._mod_quad_obj = Objective(
        expr=modified_expr, sense=solver.original_objective.sense
    )


def add_quad_terms_to_objective(
    solver: PyomoSolver,
    linvars: Var,
    quadvars: List[VarData],
    center: List[float],
    penalty: float = 1.0,
) -> None:
    add_terms_to_objective(solver, linvars)
    solver.model._mod_obj.deactivate()
    update_quad_terms_in_objective(solver, quadvars, center, penalty)

from typing import List

from pyomo.environ import Var, Objective, ScalarVar, minimize

from pyodsp.solver.pyomo_solver import PyomoSolver


def add_linear_terms_to_objective(
    solver: PyomoSolver, coeffs: List[float], vars: Var | List[ScalarVar]
) -> None:
    solver.original_objective.deactivate()
    update_linear_terms_in_objective(solver, coeffs, vars)


def update_linear_terms_in_objective(
    solver: PyomoSolver, coeffs: List[float], vars: Var | List[ScalarVar]
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
    quadvars: List[ScalarVar],
    center: List[float],
    penalty: float = 1.0,
) -> None:
    """Add a quadratic proximal penalty on top of `solver.model._mod_obj`.

    Only used by ProximalBundleMethod/RestrictedBundleMethod, whose
    `_mod_obj` is always built (via CuttingPlaneMethod.build_theta_objective)
    as a minimize-sense objective regardless of the true optimization
    direction — so the penalty is always added, never subtracted.
    """
    modified_expr = solver.model._mod_obj.expr + 0.5 * sum(
        penalty * (quadvar - val) ** 2 for quadvar, val in zip(quadvars, center)
    )

    if solver.model.component("_mod_quad_obj") is not None:
        solver.model.del_component("_mod_quad_obj")

    solver.model._mod_quad_obj = Objective(expr=modified_expr, sense=minimize)

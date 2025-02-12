from dataclasses import dataclass
from typing import List, Tuple

from pyomo.environ import (
    Var,
    Constraint,
    Objective,
    RangeSet,
    NonNegativeReals,
    inequality,
)
from pyomo.core.base.var import VarData
from pyomo.core.base.constraint import ConstraintData

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


@dataclass
class RestrictedInfo:
    constrs: List[ConstraintData]
    xlb: List[float | None]
    xub: List[float | None]


def create_restricted_mode(
    solver: PyomoSolver, constrs: List[ConstraintData]
) -> RestrictedInfo:
    xlb: List[float | None] = []
    xub: List[float | None] = []
    for var in solver.vars:
        xlb.append(var.lb)
        xub.append(var.ub)

    def constr_rule(m, i: int):
        constr = constrs[i]
        lower = None if constr.lower is None else 0
        body = constr.body
        upper = None if constr.upper is None else 0
        return inequality(lower, body, upper)

    solver.model._restricted_constrs = Constraint(
        RangeSet(0, len(constrs) - 1), rule=constr_rule
    )

    return RestrictedInfo(constrs, xlb, xub)


def activate_restricted_mode(solver: PyomoSolver, info: RestrictedInfo):
    for constr in info.constrs:
        constr.deactivate()
    solver.model._restricted_constrs.activate()
    for var in solver.vars:
        var.setlb(-1)
        var.setub(1)


def deactivate_restricted_mode(solver: PyomoSolver, info: RestrictedInfo):
    solver.model._restricted_constrs.deactivate()
    for i, var in enumerate(solver.vars):
        var.setlb(info.xlb[i])
        var.setub(info.xub[i])
    for constr in info.constrs:
        constr.activate()

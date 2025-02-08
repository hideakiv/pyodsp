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


def create_relaxed_mode(
    solver: PyomoSolver, constrs: List[ConstraintData]
) -> Tuple[Objective, Constraint]:
    solver.model.add_component(
        "_relaxed_plus",
        Var(RangeSet(0, len(constrs) - 1), domain=NonNegativeReals),
    )
    solver.model.add_component(
        "_relaxed_minus",
        Var(RangeSet(0, len(constrs) - 1), domain=NonNegativeReals),
    )

    def constr_rule(m, i: int):
        constr = constrs[i]
        lower = constr.lower
        body = constr.body
        upper = constr.upper
        modified_body = body + m._relaxed_plus[i] - m._relaxed_minus[i]
        return inequality(lower, modified_body, upper)

    solver.model._relaxed_constrs = Constraint(
        RangeSet(0, len(constrs) - 1), rule=constr_rule
    )

    solver.model._relaxed_obj = Objective(
        expr=sum(
            solver.model._relaxed_plus[i] + solver.model._relaxed_minus[i]
            for i in range(len(constrs))
        ),
        sense=solver.original_objective.sense,
    )
    return solver.model._relaxed_obj, solver.model._relaxed_constrs


def activate_relaxed_mode(solver: PyomoSolver, constrs: List[ConstraintData]):
    solver.original_objective.deactivate()
    for constr in constrs:
        constr.deactivate()

    solver.model._relaxed_obj.activate()
    solver.model._relaxed_constrs.activate()


def deactivate_relaxed_mode(solver: PyomoSolver, constrs: List[ConstraintData]):
    solver.model._relaxed_obj.deactivate()
    solver.model._relaxed_constrs.deactivate()

    solver.original_objective.activate()
    for constr in constrs:
        constr.activate()


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

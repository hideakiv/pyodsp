from typing import List, Dict
from dataclasses import dataclass

from pyomo.environ import ConcreteModel, Constraint
from pyomo.core.base.var import VarData
from pyomo.core.base.constraint import ConstraintData, IndexedConstraint
from pyomo.repn import generate_standard_repn


@dataclass
class CouplingData:
    """Coefficients of the variables in the constraint."""

    constraint: ConstraintData
    coefficients: Dict[int, float]
    vars: List[VarData]


def get_nonzero_coefficients_from_model(
    model: ConcreteModel, vars: List[VarData]
) -> List[CouplingData]:
    """Get the nonzero coefficients of the variables in the constraints.

    Args:
        model: The Pyomo model.
        vars: The variables to get the coefficients of.

    Returns:
        A tuple of the constraints and the coefficients.
    """
    coupling_list: List[CouplingData] = []
    for constraint in model.component_objects(ctype=Constraint):
        if isinstance(constraint, ConstraintData):
            coupling_data = get_nonzero_coefficients_from_constraint(constraint, vars)
            if len(coupling_data.coefficients) > 0:
                coupling_list.append(coupling_data)
        elif isinstance(constraint, IndexedConstraint):
            for index in constraint:
                coupling_data = get_nonzero_coefficients_from_constraint(
                    constraint[index], vars
                )
                if len(coupling_data.coefficients) > 0:
                    coupling_list.append(coupling_data)
    return coupling_list


def get_nonzero_coefficients_from_constraint(
    constraint: ConstraintData, vars: List[VarData]
) -> CouplingData:
    """Get the nonzero coefficients of the variables in the constraint.

    Args:
        constraint: The constraint.
        vars: The variables to get the coefficients of.

    Returns:
        A tuple of the constraint and the coefficients.
    """
    repn = generate_standard_repn(constraint.body)
    all_coefficients = {
        var.name: coef for var, coef in zip(repn.linear_vars, repn.linear_coefs)
    }
    coefficients: Dict[int, float] = {}
    for i, var in enumerate(vars):
        if var.name in all_coefficients:
            coefficients[i] = all_coefficients[var.name]
    return CouplingData(constraint, coefficients, vars)

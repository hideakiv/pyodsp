from calendar import c
from typing import List, Dict, Tuple
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


@dataclass
class LagrangianData:
    """Data for coupling constraints"""

    matrix: Dict[int, List[Dict[int, float]]]
    rhs: List[float]
    sense: List[int]  # leq: 1, eq: 0, geq: -1
    constraints: List[ConstraintData]
    vars_dict: Dict[int, List[VarData]]


def get_nonzero_coefficients_group(
    model: ConcreteModel, vars_dict: Dict[int, List[VarData]]
) -> LagrangianData:
    matrix: Dict[int, List[Dict[int, float]]] = {}
    rhss: List[float] = []
    senses: List[int] = []
    constraints: List[ConstraintData] = []
    for key in vars_dict.keys():
        matrix[key] = []
    for constraint in model.component_objects(ctype=Constraint):
        if isinstance(constraint, ConstraintData):
            coupling_data, rhs, sense = get_nonzero_coefficients_group_from_constraint(
                constraint, vars_dict
            )
            for key, var in coupling_data.items():
                matrix[key].append(var.coefficients)
            rhss.append(rhs)
            senses.append(sense)
            constraints.append(constraint)
        elif isinstance(constraint, IndexedConstraint):
            for index in constraint:
                coupling_data, rhs, sense = (
                    get_nonzero_coefficients_group_from_constraint(
                        constraint[index], vars_dict
                    )
                )
                for key, var in coupling_data.items():
                    matrix[key].append(var.coefficients)
                rhss.append(rhs)
                senses.append(sense)
                constraints.append(constraint[index])
    return LagrangianData(matrix, rhss, senses, constraints, vars_dict)


def get_nonzero_coefficients_group_from_constraint(
    constraint: ConstraintData, vars_dict: Dict[int, List[VarData]]
) -> Tuple[Dict[int, CouplingData], float, int]:
    repn = generate_standard_repn(constraint.body)
    all_coefficients = {
        var.name: coef for var, coef in zip(repn.linear_vars, repn.linear_coefs)
    }
    coupling_data: Dict[int, CouplingData] = {}
    for key, vars in vars_dict.items():
        coefficients: Dict[int, float] = {}
        for i, var in enumerate(vars):
            if var.name in all_coefficients:
                coefficients[i] = all_coefficients[var.name]
        coupling_data[key] = CouplingData(constraint, coefficients, vars)
    rhs = repn

    # get sense of constraint TODO verify
    rhs: float = 0.0
    sense: int = 0
    if constraint.equality:
        rhs = constraint.ub
    else:
        if constraint.has_ub():
            rhs = constraint.ub
            sense = 1
        else:
            rhs = constraint.lb
            sense = -1
    return coupling_data, rhs, sense

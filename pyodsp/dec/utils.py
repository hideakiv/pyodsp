from typing import List, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass

from pyomo.environ import ConcreteModel, Constraint, ScalarVar
from pyomo.core.base.constraint import ScalarConstraint
from pyomo.repn import generate_standard_repn


def create_directory(filedir: Path) -> None:
    try:
        filedir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"An error occurred: {e}")


@dataclass
class CouplingData:
    """Coefficients of the variables in the constraint."""

    constraint: ScalarConstraint
    coefficients: Dict[int, float]
    vars: List[ScalarVar]


def get_nonzero_coefficients_from_model(
    model: ConcreteModel, vars: List[ScalarVar]
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
        if isinstance(constraint, ScalarConstraint):
            coupling_data = get_nonzero_coefficients_from_constraint(constraint, vars)
            if len(coupling_data.coefficients) > 0:
                coupling_list.append(coupling_data)
        else:
            for index in constraint:
                coupling_data = get_nonzero_coefficients_from_constraint(
                    constraint[index], vars
                )
                if len(coupling_data.coefficients) > 0:
                    coupling_list.append(coupling_data)
    return coupling_list


def get_nonzero_coefficients_from_constraint(
    constraint: ScalarConstraint, vars: List[ScalarVar]
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


SparseMatrix = List[Dict[int, float]]


@dataclass
class LagrangianData:
    """Data for coupling constraints"""

    lbs: List[float | None]
    matrix: Dict[int, SparseMatrix]
    ubs: List[float | None]
    constraints: List[ScalarConstraint]
    vars_dict: Dict[int, List[ScalarVar]]


def get_nonzero_coefficients_group(
    model: ConcreteModel, vars_dict: Dict[int, List[ScalarVar]]
) -> LagrangianData:
    matrix: Dict[int, SparseMatrix] = {}
    lbs: List[float | None] = []
    ubs: List[float | None] = []
    constraints: List[ScalarConstraint] = []
    for key in vars_dict.keys():
        matrix[key] = []
    for constraint in model.component_objects(ctype=Constraint):
        if isinstance(constraint, ScalarConstraint):
            lb, coupling_data, ub = get_nonzero_coefficients_group_from_constraint(
                constraint, vars_dict
            )
            for key, var in coupling_data.items():
                matrix[key].append(var.coefficients)
            lbs.append(lb)
            ubs.append(ub)
            constraints.append(constraint)
        else:
            for index in constraint:
                lb, coupling_data, ub = get_nonzero_coefficients_group_from_constraint(
                    constraint[index], vars_dict
                )
                for key, var in coupling_data.items():
                    matrix[key].append(var.coefficients)
                lbs.append(lb)
                ubs.append(ub)
                constraints.append(constraint[index])
    return LagrangianData(lbs, matrix, ubs, constraints, vars_dict)


def get_nonzero_coefficients_group_from_constraint(
    constraint: ScalarConstraint, vars_dict: Dict[int, List[ScalarVar]]
) -> Tuple[float | None, Dict[int, CouplingData], float | None]:
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

    return constraint.lb, coupling_data, constraint.ub

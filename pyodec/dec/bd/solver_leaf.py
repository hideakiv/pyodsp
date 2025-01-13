from typing import List

from pyomo.environ import (
    ConcreteModel,
    Suffix,
    Var,
    Constraint,
    Objective,
    NonNegativeReals,
    RangeSet,
    inequality,
)
from pyomo.core.base.var import VarData
from pyomo.core.base.constraint import ConstraintData

from .solver import BdSolver
from .cuts import Cut, OptimalityCut, FeasibilityCut
from pyodec.dec.utils import CouplingData, get_nonzero_coefficients_from_model

solver_dual_sign_convention = dict()
solver_dual_sign_convention["ipopt"] = -1
solver_dual_sign_convention["gurobi"] = -1
solver_dual_sign_convention["gurobi_direct"] = -1
solver_dual_sign_convention["gurobi_persistent"] = -1
solver_dual_sign_convention["cplex"] = -1
solver_dual_sign_convention["cplex_direct"] = -1
solver_dual_sign_convention["cplexdirect"] = -1
solver_dual_sign_convention["cplex_persistent"] = -1
solver_dual_sign_convention["glpk"] = -1
solver_dual_sign_convention["cbc"] = -1
solver_dual_sign_convention["xpress_direct"] = -1
solver_dual_sign_convention["xpress_persistent"] = -1
solver_dual_sign_convention["appsi_highs"] = 1


class BdSolverLeaf(BdSolver):
    def __init__(self, model: ConcreteModel, solver: str, **kwargs):
        super().__init__(model, solver, **kwargs)

        self.model.dual = Suffix(direction=Suffix.IMPORT)
        self.sign_convention = solver_dual_sign_convention[solver]

        self.coupling_values: List[float] = None
        self.coupling_info: List[CouplingData] = None
        self.coupling_constraints: List[ConstraintData] = None

        self.enfeasibled_constraints: List[ConstraintData] = []
        self.enfeasibled_objective: Objective = None

    def build(self, coupling_vars: List[VarData]) -> None:
        self.coupling_info = get_nonzero_coefficients_from_model(
            self.model, coupling_vars
        )
        self.coupling_constraints = [
            coupling_data.constraint for coupling_data in self.coupling_info
        ]

    def fix_variables(self, vars: List[VarData], values: List[float]) -> None:
        """Fix the variables to a specified value

        Args:
            vars: The variables to be fixed.
            values: The values to be set.
        """
        self.coupling_values = values
        for i, var in enumerate(vars):
            var.fix(values[i])

    def get_subgradient(self, coupling_vars: List[VarData]) -> Cut:
        if self.is_optimal():
            return self._optimality_cut(coupling_vars)
        elif self.is_infeasible():
            return self._feasibility_cut(coupling_vars)
        else:
            raise ValueError("Unknown solver status")

    def _optimality_cut(self, coupling_vars: List[VarData]) -> OptimalityCut:
        pi = [
            self.sign_convention * self.model.dual[constr]
            for constr in self.coupling_constraints
        ]
        objective = self.get_objective_value()
        coef = [0 for _ in range(len(coupling_vars))]
        constant = objective
        for i, dual_var in enumerate(pi):
            coupling_data = self.coupling_info[i]
            for j, coefficients in coupling_data.coefficients.items():
                temp = dual_var * coefficients
                coef[j] += temp
                constant += temp * self.coupling_values[j]
        return OptimalityCut(
            coefficients=coef,
            constant=constant,
            objective_value=self.get_objective_value(),
        )

    def _feasibility_cut(self, coupling_vars: List[VarData]) -> FeasibilityCut:
        if self.enfeasibled_objective is None:
            self._create_feasible_mode(self.coupling_constraints)

        self._activate_feasible_mode()
        self.solve()

        sigma = [
            self.sign_convention * self.model.dual[constr]
            for constr in self.enfeasibled_constraints
        ]
        objective = self.get_objective_value()

        self._deactivate_feasible_mode()

        coef = [0 for _ in range(len(coupling_vars))]
        constant = objective
        for i, dual_ray in enumerate(sigma):
            coupling_data = self.coupling_info[i]
            for j, coefficients in coupling_data.coefficients.items():
                temp = dual_ray * coefficients
                coef[j] += temp
                constant += temp * self.coupling_values[j]
        return FeasibilityCut(
            coefficients=coef,
            constant=constant,
        )

    def _create_feasible_mode(self, constrs: List[ConstraintData]) -> None:
        self.model.add_component(
            "_v_feas_plus",
            Var(RangeSet(0, len(constrs) - 1), domain=NonNegativeReals),
        )
        self.model.add_component(
            "_v_feas_minus",
            Var(RangeSet(0, len(constrs) - 1), domain=NonNegativeReals),
        )
        for i, constr in enumerate(constrs):
            lower = constr.lower
            body = constr.body
            upper = constr.upper
            modified_body = (
                body + self.model._v_feas_plus[i] - self.model._v_feas_minus[i]
            )
            self.model.add_component(
                f"_c_feas_{i}", Constraint(expr=inequality(lower, modified_body, upper))
            )
            self.enfeasibled_constraints.append(self.model.component(f"_c_feas_{i}"))

        self.model._enf_obj = Objective(
            expr=sum(
                self.model._v_feas_plus[i] + self.model._v_feas_minus[i]
                for i in range(len(constrs))
            ),
            sense=self.original_objective.sense,
        )
        self.enfeasibled_objective = self.model._enf_obj

    def _activate_feasible_mode(self) -> None:
        self.original_objective.deactivate()
        for constr in self.coupling_constraints:
            constr.deactivate()

        self.model._enf_obj.activate()
        for constr in self.enfeasibled_constraints:
            constr.activate()

    def _deactivate_feasible_mode(self) -> None:
        self.model._enf_obj.deactivate()
        for constr in self.enfeasibled_constraints:
            constr.deactivate()

        self.original_objective.activate()
        for constr in self.coupling_constraints:
            constr.activate()

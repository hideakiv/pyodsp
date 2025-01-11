from typing import List
from pyomo.environ import (
    ConcreteModel,
    SolverFactory,
    Suffix,
    Objective,
    value,
    Var,
    Constraint,
    NonNegativeReals,
    RangeSet,
)
from pyomo.core.base.var import VarData
from pyomo.core.base.constraint import ConstraintData
from pyomo.opt import TerminationCondition

from .subsolver import SubSolver

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


class PyomoSubSolver(SubSolver):
    """Subsolver using Pyomo's solver interface."""

    def __init__(self, model: ConcreteModel, solver: str, **kwargs) -> None:
        """Initialize the subsolver.

        Args:
            solver: The solver to use.
        """
        self.solver = SolverFactory(solver)
        self.model = model
        self.use_dual = kwargs.pop("use_dual", False)
        self.solver_kwargs = kwargs

        self.original_objective = self.get_objective()

        if self.use_dual:
            if solver not in solver_dual_sign_convention:
                raise ValueError("Solver not supported for dual")
            self.model.dual = Suffix(direction=Suffix.IMPORT)
            self.enfeasibled_constraints = []

        self.sign_convention = solver_dual_sign_convention[solver]
        self.results = None

    def solve(self) -> None:
        """Solve the model."""

        self.results = self.solver.solve(self.model, **self.solver_kwargs)

    def get_objective(self):
        """Get the objective of the model"""
        for obj in self.model.component_objects(Objective, active=True):
            # There should be only one objective
            return obj

    def get_objective_value(self) -> float:
        """Get the objective value of the model"""
        return value(self.get_objective())

    def get_solution(self, vars: List[VarData]) -> List[float]:
        """Get the solution of the model.

        Args:
            vars: The variables to get the solution of.
        """
        return [var.value for var in vars]

    def get_dual_solution(self, constrs: List[ConstraintData]) -> List[float]:
        """Get the dual solution of the model.

        Args:
            constrs: The constraints to get the dual solution of.
        """
        if not self.use_dual:
            raise ValueError("Cannot access to dual")

        return [self.sign_convention * self.model.dual[constr] for constr in constrs]

    def get_dual_ray(self, constrs: List[ConstraintData]) -> List[float]:
        """Get the dual ray of the model.

        Args:
            constrs: The constraints to get the dual ray of.
        """
        if not self.use_dual:
            raise ValueError("Cannot access to dual")

        if len(self.enfeasibled_constraints) == 0:
            self._create_enfeasibled_constraints(constrs)

        return [self.sign_convention * self.model.dual[constr] for constr in constrs]

    def _create_enfeasibled_constraints(self, constrs: List[ConstraintData]) -> None:
        self.model.add_component(
            "_v_feas_plus",
            Var(RangeSet(0, len(constrs) - 1), domain=NonNegativeReals),
        )
        self.model.add_component(
            "_v_feas_minus",
            Var(RangeSet(0, len(constrs) - 1), domain=NonNegativeReals),
        )
        for i, constr in enumerate(constrs):
            new_constr = (
                constr.expr + self.model._v_feas_plus[i] - self.model._v_feas_minus[i]
            )
            self.model.add_component(f"_c_feas_{i}", Constraint(expr=new_constr))
            self.enfeasibled_constraints.append(self.model.component(f"_c_feas_{i}"))

    def fix_variables(self, vars: List[VarData], values: List[float]) -> None:
        """Fix the variables to a specified value

        Args:
            vars: The variables to be fixed.
            values: The values to be set.
        """
        for i, var in enumerate(vars):
            var.fix(values[i])

    def is_optimal(self) -> bool:
        """Returns whether the model is optimal."""
        return self.results.solver.termination_condition == TerminationCondition.optimal

    def is_infeasible(self) -> bool:
        """Returns whether the model is infeasible."""
        return (
            self.results.solver.termination_condition == TerminationCondition.infeasible
        )

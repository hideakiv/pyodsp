from typing import List
from pyomo.environ import ConcreteModel, SolverFactory, Suffix, Objective, value
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

        if self.use_dual:
            if solver not in solver_dual_sign_convention:
                raise ValueError("Solver not supported for dual")
            self.model.dual = Suffix(direction=Suffix.IMPORT)

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

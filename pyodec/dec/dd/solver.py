from typing import List
from pyomo.environ import (
    ConcreteModel,
    SolverFactory,
    Objective,
    value,
)
from pyomo.core.base.var import VarData
from pyomo.opt import TerminationCondition


class DdSolver:
    """Base class for Dual Decomposition subproblem solvers"""

    def __init__(self, model: ConcreteModel, solver: str, **kwargs):
        """Initialize the subsolver.

        Args:
            model: The Pyomo model.
            solver: The solver to use.
        """
        self.solver = SolverFactory(solver)
        self.model = model
        self._solver_kwargs = kwargs

        self.original_objective = self.get_objective()

        self._results = None

    def solve(self) -> None:
        """Solve the model."""

        self._results = self.solver.solve(
            self.model, load_solutions=False, **self._solver_kwargs
        )
        if self.is_optimal():
            self.model.solutions.load_from(self._results)

    def get_objective(self) -> Objective:
        """Get the objective of the model"""
        for obj in self.model.component_objects(Objective, active=True):
            # There should be only one objective
            return obj
        else:
            raise ValueError("Objective not found")

    def get_objective_value(self) -> float:
        """Get the objective value of the model"""
        return value(self.get_objective())

    def get_objective_sense(self) -> bool:
        """Get the sense of the objective.

        Returns:
            True if the objective is to be minimized, False otherwise.
        """
        return self.get_objective().sense > 0

    def get_solution(self, vars: List[VarData]) -> List[float]:
        """Get the solution of the model.

        Args:
            vars: The variables to get the solution of.
        """
        return [var.value for var in vars]

    def is_optimal(self) -> bool:
        """Returns whether the model is optimal."""
        return (
            self._results.solver.termination_condition == TerminationCondition.optimal
        )

    def is_infeasible(self) -> bool:
        """Returns whether the model is infeasible."""
        return (
            self._results.solver.termination_condition
            == TerminationCondition.infeasible
        )

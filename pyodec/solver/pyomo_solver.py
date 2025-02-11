from typing import List
from pathlib import Path

import pandas as pd

from pyomo.environ import (
    ConcreteModel,
    Var,
    SolverFactory,
    Objective,
    value,
)
from pyomo.core.base.var import VarData
from pyomo.opt import TerminationCondition

from .solver import Solver

solver_dual_sign_convention = dict()
solver_dual_sign_convention["ipopt"] = 1
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


class PyomoSolver(Solver):
    """Base class for solvers using Pyomo"""

    def __init__(
        self, model: ConcreteModel, solver: str, vars: List[VarData], **kwargs
    ):
        """Initialize the subsolver.

        Args:
            model: The Pyomo model.
            solver: The solver to use.
            vars: The variables in focus
        """
        self.solver = SolverFactory(solver)
        self.model = model
        self.vars = vars
        self._solver_kwargs = kwargs

        self.original_objective = self._get_objective()
        self.sign_convention = solver_dual_sign_convention[solver]

        self._results = None

    def solve(self) -> None:
        """Solve the model."""

        self._results = self.solver.solve(
            self.model, load_solutions=False, **self._solver_kwargs
        )
        if self.is_optimal():
            self.model.solutions.load_from(self._results)

    def _get_objective(self) -> Objective:
        """Get the objective of the model"""
        for obj in self.model.component_objects(Objective, active=True):
            # There should be only one objective
            return obj
        else:
            raise ValueError("Objective not found")

    def get_objective_value(self) -> float:
        """Get the objective value of the model"""
        return value(self._get_objective())

    def get_original_objective_value(self) -> float:
        """Get the objective value of the model"""
        return value(self.original_objective)

    def is_minimize(self) -> bool:
        """Get the sense of the objective.

        Returns:
            True if the objective is to be minimized, False otherwise.
        """
        return self._get_objective().sense > 0

    def get_solution(self) -> List[float]:
        """Get the solution of the model."""
        return [var.value for var in self.vars]

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

    def is_unbounded(self) -> bool:
        """Returns whether the model is unbounded."""
        return (
            self._results.solver.termination_condition == TerminationCondition.unbounded
        )

    def save(self, dir: Path) -> None:
        """outputs solution to dir"""
        path = dir / "sol.csv"
        solution = {}
        for v in self.model.component_objects(Var, active=True):
            if str(v) == "_relaxed_minus" or str(v) == "_relaxed_plus":
                continue
            varobject = getattr(self.model, str(v))
            for index in varobject:
                if index is None:
                    varname = str(v)
                else:
                    varname = f"{v}_{index}"
                solution[varname] = varobject[index].value

        # Convert the solution to a DataFrame
        sol = pd.DataFrame(list(solution.items()), columns=["var", "val"])

        sol.to_csv(path, sep="\t", index=False)

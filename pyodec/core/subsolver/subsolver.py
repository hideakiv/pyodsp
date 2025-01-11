from abc import ABC, abstractmethod
from typing import List

from pyomo.core.base.var import VarData
from pyomo.core.base.constraint import ConstraintData


class SubSolver(ABC):
    """Base class for subsolvers."""

    @abstractmethod
    def solve(self) -> None:
        """Solve the model."""
        pass

    @abstractmethod
    def get_objective(self):
        """Get the objective of the model"""
        pass

    @abstractmethod
    def get_objective_value(self) -> float:
        """Get the objective value of the model"""
        pass

    @abstractmethod
    def get_solution(self, vars: List[VarData]) -> List[float]:
        """Get the solution of the model.

        Args:
            vars: The variables to get the solution of.
        """
        pass

    @abstractmethod
    def get_dual_solution(self, constrs: List[ConstraintData]) -> List[float]:
        """Get the dual solution of the model.

        Args:
            constrs: The constraints to get the dual solution of.
        """
        pass

    @abstractmethod
    def get_dual_ray(self, constrs: List[ConstraintData]) -> List[float]:
        """Get the dual ray of the model.

        Args:
            constrs: The constraints to get the dual ray of.
        """
        pass

    @abstractmethod
    def fix_variables(self, vars: List[VarData], values: List[float]) -> None:
        """Fix the variables to a specified value

        Args:
            vars: The variables to be fixed.
            values: The values to be set.
        """
        pass

    @abstractmethod
    def is_optimal(self) -> bool:
        """Returns whether the model is optimal."""
        pass

    @abstractmethod
    def is_infeasible(self) -> bool:
        """Returns whether the model is infeasible."""
        pass

from abc import ABC, abstractmethod
from typing import List


class Solver(ABC):
    """Base class for Dual Decomposition subproblem solvers"""

    @abstractmethod
    def solve(self) -> None:
        """Solve the model."""
        pass

    @abstractmethod
    def get_objective_value(self) -> float:
        """Get the objective value of the model"""
        return 0.0

    @abstractmethod
    def is_minimize(self) -> bool:
        """Get the sense of the objective.

        Returns:
            True if the objective is to be minimized, False otherwise.
        """
        return True

    @abstractmethod
    def get_solution(self) -> List[float]:
        """Get the solution of the model."""
        return []

    @abstractmethod
    def is_optimal(self) -> bool:
        """Returns whether the model is optimal."""
        return False

    @abstractmethod
    def is_infeasible(self) -> bool:
        """Returns whether the model is infeasible."""
        return False

    @abstractmethod
    def is_unbounded(self) -> bool:
        """Returns whether the model is unbounded."""
        return False

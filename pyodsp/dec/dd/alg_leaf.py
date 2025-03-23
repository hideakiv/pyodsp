from abc import ABC, abstractmethod
from typing import List, Tuple

from ..node._alg import IAlgLeaf

class DdAlgLeaf(IAlgLeaf, ABC):

    @abstractmethod
    def build(self) -> None:
        pass

    @abstractmethod
    def update_objective(self, coeffs: List[float]) -> None:
        pass

    @abstractmethod
    def get_solution_or_ray(self) -> Tuple[bool, List[float], float]:
        pass

    @abstractmethod
    def get_len_vars(self) -> int:
        pass

    @abstractmethod
    def get_objective_value(self) -> float:
        pass

    @abstractmethod
    def fix_variables_and_solve(self, values: List[float]) -> None:
        pass

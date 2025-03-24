from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

from ..node._alg import IAlgLeaf

class DdAlgLeaf(IAlgLeaf, ABC):

    @abstractmethod
    def set_coupling_matrix(self, coupling_matrix: List[Dict[int, float]]) -> None:
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

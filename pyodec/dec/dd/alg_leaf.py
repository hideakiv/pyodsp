from abc import ABC, abstractmethod
from typing import List, Tuple


class DdAlgLeaf(ABC):

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
    def is_minimize(self) -> bool:
        pass

    @abstractmethod
    def get_len_vars(self) -> int:
        pass

    @abstractmethod
    def get_objective_value(self) -> float:
        pass

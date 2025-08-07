from abc import ABC, abstractmethod
from typing import Any


class ILogger(ABC):
    @abstractmethod
    def log_initialization(self, **kwargs):
        pass

    @abstractmethod
    def log_master_problem(self, iteration: int, objective_value: float, x: Any):
        pass

    @abstractmethod
    def log_sub_problem(self, idx, cut_type: str, coefficients: Any, constant: float):
        pass

    @abstractmethod
    def log_finaliziation(self):
        pass

    @abstractmethod
    def log_completion(self, objective_value: float | None):
        pass

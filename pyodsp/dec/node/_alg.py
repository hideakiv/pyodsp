from abc import ABC, abstractmethod
from typing import List, Tuple
from pathlib import Path

from pyodsp.alg.cuts import Cut, CutList

from ..run._message import NodeIdx, IMessage

class IAlg(ABC):
    @abstractmethod
    def save(self, dir: Path) -> None:
        pass

    @abstractmethod
    def is_minimize(self) -> bool:
        pass

class IAlgRoot(IAlg, ABC):
    @abstractmethod
    def set_logger(self, idx: NodeIdx, depth: int) -> None:
        pass

    @abstractmethod
    def build(self, bounds: List[float | None]) -> None:
        pass

    @abstractmethod
    def run_step(self, cuts_list: List[CutList] | None) -> Tuple[int, List[float]]:
        pass

    @abstractmethod
    def get_init_message(self, **kwargs) -> IMessage:
        pass

    @abstractmethod
    def add_cuts(self, cuts_list: List[CutList]) -> None:
        pass

    @abstractmethod
    def reset_iteration(self) -> None:
        pass

    @abstractmethod
    def get_solution_dn(self) -> List[float]:
        pass

    @abstractmethod
    def get_num_vars(self) -> int:
        pass

class IAlgLeaf(IAlg, ABC):
    @abstractmethod
    def build(self) -> None:
        pass

    @abstractmethod
    def get_objective_value(self) -> float:
        pass

    @abstractmethod
    def pass_init_message(self, message: IMessage) -> None:
        pass

    @abstractmethod
    def pass_solution(self, solution: List[float]) -> None:
        pass

    @abstractmethod
    def pass_final_message(self, message: IMessage) -> None:
        pass

    @abstractmethod
    def get_subgradient(self) -> Cut:
        pass
from abc import ABC, abstractmethod
from typing import List, Tuple
from pathlib import Path

from pyodsp.alg.cuts import Cut, CutList

from ..run._message import NodeIdx, InitMessage, FinalMessage, DnMessage, UpMessage


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
    def run_step(self, cuts_list: List[CutList] | None) -> Tuple[int, DnMessage]:
        pass

    @abstractmethod
    def get_init_message(self, **kwargs) -> InitMessage:
        pass

    @abstractmethod
    def add_cuts(self, cuts_list: List[CutList]) -> None:
        pass

    @abstractmethod
    def reset_iteration(self) -> None:
        pass

    @abstractmethod
    def get_dn_message(self) -> DnMessage:
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
    def pass_init_message(self, message: InitMessage) -> None:
        pass

    @abstractmethod
    def pass_dn_message(self, message: DnMessage) -> None:
        pass

    @abstractmethod
    def pass_final_message(self, message: FinalMessage) -> None:
        pass

    @abstractmethod
    def get_subgradient(self) -> Cut:
        pass

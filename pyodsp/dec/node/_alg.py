from abc import ABC, abstractmethod
from typing import List, Tuple
from pathlib import Path

from pyodsp.alg.bm.cuts import CutList

from ._message import (
    NodeIdx,
    InitDnMessage,
    InitUpMessage,
    FinalDnMessage,
    FinalUpMessage,
    DnMessage,
    UpMessage,
)


class IAlg(ABC):
    @abstractmethod
    def save(self, dir: Path) -> None:
        pass

    @abstractmethod
    def is_minimize(self) -> bool:
        pass


class IAlgRoot(IAlg, ABC):
    @abstractmethod
    def set_logger(self, idx: NodeIdx, depth: int, level: int) -> None:
        pass

    @abstractmethod
    def build(self, bounds: List[float | None]) -> None:
        pass

    @abstractmethod
    def run_step(self, cuts_list: List[CutList] | None) -> Tuple[int, DnMessage]:
        pass

    @abstractmethod
    def get_init_dn_message(self, **kwargs) -> InitDnMessage:
        pass

    @abstractmethod
    def add_cuts(self, cuts_list: List[CutList]) -> None:
        pass

    @abstractmethod
    def reset_iteration(self) -> None:
        pass

    @abstractmethod
    def get_final_dn_message(self, **kwargs) -> FinalDnMessage:
        pass

    @abstractmethod
    def pass_final_up_message(self, children_obj: float | None) -> FinalUpMessage:
        pass

    @abstractmethod
    def get_num_vars(self) -> int:
        pass


class IAlgLeaf(IAlg, ABC):
    @abstractmethod
    def build(self) -> None:
        pass

    @abstractmethod
    def pass_init_dn_message(self, message: InitDnMessage) -> None:
        pass

    @abstractmethod
    def get_init_up_message(self) -> InitUpMessage:
        pass

    @abstractmethod
    def pass_dn_message(self, message: DnMessage) -> None:
        pass

    @abstractmethod
    def pass_final_dn_message(self, message: FinalDnMessage) -> None:
        pass

    @abstractmethod
    def get_final_up_message(self) -> FinalUpMessage:
        pass

    @abstractmethod
    def get_up_message(self) -> UpMessage:
        pass

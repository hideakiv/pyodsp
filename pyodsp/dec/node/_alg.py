from abc import ABC, abstractmethod
from typing import List, Tuple
from pathlib import Path

from pyodsp.alg.cuts import CutList

from ._node import NodeIdx

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
    def add_cuts(self, cuts_list: List[CutList]) -> None:
        pass

class IAlgLeaf(IAlg, ABC):
    @abstractmethod
    def build(self) -> None:
        pass
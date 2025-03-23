from abc import ABC, abstractmethod
from pathlib import Path

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

class IAlgLeaf(IAlg, ABC):
    pass
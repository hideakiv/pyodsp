from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Dict, Tuple

from pyodsp.alg.cuts import Cut

NodeIdx = Any

class INode(ABC):
    @abstractmethod
    def get_idx(self) -> NodeIdx:
        pass

    @abstractmethod
    def get_depth(self) -> int:
        pass

    @abstractmethod
    def set_depth(self, depth: int) -> None:
        pass

    @abstractmethod
    def get_parents(self) -> List[NodeIdx]:
        pass

    @abstractmethod
    def get_children(self) -> List[NodeIdx]:
        pass

    @abstractmethod
    def build(self) -> None:
        pass

    @abstractmethod
    def save(self, dir: Path) -> None:
        pass

    @abstractmethod
    def is_minimize(self) -> bool:
        pass

class INodeParent(INode, ABC):
    @abstractmethod
    def add_child(self, idx: NodeIdx, multiplier: float) -> None:
        pass
    
    @abstractmethod
    def set_groups(self, groups: List[List[NodeIdx]]) -> None:
        pass

    @abstractmethod
    def set_multiplier(self, idx: NodeIdx, multiplier: float) -> None:
        pass

    @abstractmethod
    def get_multiplier(self, idx: NodeIdx) -> float:
        pass

    @abstractmethod
    def set_bound(self, idx: NodeIdx, bound: float) -> None:
        pass

    @abstractmethod
    def get_bound(self, idx: NodeIdx) -> float:
        pass

    @abstractmethod
    def set_logger(self) -> None:
        pass

    @abstractmethod
    def run_step(self, cuts: Dict[int, Cut] | None) -> Tuple[int, List[float]]:
        pass

INodeRoot = INodeParent

class INodeChild(INode, ABC):
    @abstractmethod
    def add_parent(self, idx: NodeIdx) -> None:
        pass

INodeLeaf = INodeChild

class INodeInner(INodeParent, INodeChild, ABC):
    pass

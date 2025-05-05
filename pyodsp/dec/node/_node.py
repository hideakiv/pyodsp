from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Dict, Tuple

from ._alg import IAlgRoot, IAlgLeaf
from ..run._message import NodeIdx, InitMessage, FinalMessage, DnMessage, UpMessage


class INode(ABC):
    @abstractmethod
    def get_kwargs(self) -> Dict[str, Any]:
        pass

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


class INodeParent(INode, ABC):
    @abstractmethod
    def get_alg_root(self) -> IAlgRoot:
        pass

    @abstractmethod
    def add_child(self, idx: NodeIdx, multiplier: float) -> None:
        pass

    @abstractmethod
    def set_groups(self, groups: List[List[NodeIdx]]) -> None:
        pass

    @abstractmethod
    def get_groups(self) -> List[List[NodeIdx]]:
        pass

    @abstractmethod
    def set_multiplier(self, idx: NodeIdx, multiplier: float) -> None:
        pass

    @abstractmethod
    def get_multiplier(self, idx: NodeIdx) -> float:
        pass

    @abstractmethod
    def set_child_bound(self, idx: NodeIdx, bound: float) -> None:
        pass

    @abstractmethod
    def get_child_bound(self, idx: NodeIdx) -> float:
        pass

    @abstractmethod
    def set_logger(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def run_step(
        self, up_messages: Dict[NodeIdx, UpMessage] | None
    ) -> Tuple[int, DnMessage]:
        pass

    @abstractmethod
    def get_init_message(self, **kwargs) -> InitMessage:
        pass

    @abstractmethod
    def add_cuts(self, cuts: Dict[int, UpMessage]) -> None:
        pass

    @abstractmethod
    def get_final_message(self) -> FinalMessage:
        pass

    @abstractmethod
    def get_num_vars(self) -> int:
        pass


INodeRoot = INodeParent


class INodeChild(INode, ABC):
    @abstractmethod
    def get_alg_leaf(self) -> IAlgLeaf:
        pass

    @abstractmethod
    def add_parent(self, idx: NodeIdx) -> None:
        pass

    @abstractmethod
    def set_bound(self, bound: float) -> None:
        pass

    @abstractmethod
    def get_bound(self) -> float:
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
    def get_up_message(self) -> UpMessage:
        pass

    @abstractmethod
    def solve(self, message: DnMessage) -> UpMessage:
        pass


INodeLeaf = INodeChild


class INodeInner(INodeParent, INodeChild, ABC):
    pass

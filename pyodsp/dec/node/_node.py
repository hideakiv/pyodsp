from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple

from ._alg import IAlgRoot, IAlgLeaf
from ._message import (
    NodeIdx,
    InitDnMessage,
    InitUpMessage,
    FinalDnMessage,
    FinalUpMessage,
    DnMessage,
    UpMessage,
)


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
    def get_init_dn_message(self, **kwargs) -> InitDnMessage:
        pass

    @abstractmethod
    def pass_init_up_messages(self, messages: Dict[NodeIdx, InitUpMessage]) -> None:
        pass

    @abstractmethod
    def add_cuts(self, cuts: Dict[int, UpMessage]) -> None:
        pass

    @abstractmethod
    def get_final_dn_message(self, **kwargs) -> FinalDnMessage:
        pass

    @abstractmethod
    def pass_final_up_message(
        self, messages: Dict[NodeIdx, FinalUpMessage]
    ) -> FinalUpMessage:
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
    def get_bound(self) -> float | None:
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
    def get_up_message(self) -> UpMessage:
        pass

    @abstractmethod
    def pass_final_dn_message(self, message: FinalDnMessage) -> None:
        pass

    @abstractmethod
    def get_final_up_message(self) -> FinalUpMessage:
        pass

    @abstractmethod
    def solve(self, message: DnMessage) -> UpMessage:
        pass


INodeLeaf = INodeChild


class INodeInner(INodeParent, INodeChild, ABC):
    pass

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from pathlib import Path
import logging

from ._node import NodeIdx, INode, INodeParent, INodeChild, INodeInner
from ._alg import IAlgRoot, IAlgLeaf
from .cut_aggregator import CutAggregator
from ._message import (
    InitDnMessage,
    InitUpMessage,
    FinalDnMessage,
    FinalUpMessage,
    DnMessage,
    UpMessage,
)
from ..utils import create_directory


class DecNode(INode, ABC):
    def __init__(self, idx: NodeIdx, **kwargs) -> None:
        self.idx: NodeIdx = idx
        self.depth: int | None = None

        self.parents: List[NodeIdx] = []
        self.children: List[NodeIdx] = []
        self.children_multipliers: Dict[NodeIdx, float] = {}
        self.children_bounds: Dict[int, float] = {}
        self.groups = []

        self.built = False

    def get_idx(self) -> NodeIdx:
        return self.idx

    def get_depth(self) -> int:
        if self.depth is None:
            raise ValueError("Depth not initialized")
        return self.depth

    def set_depth(self, depth: int) -> None:
        self.depth = depth

    def get_parents(self) -> List[NodeIdx]:
        return self.parents

    def get_children(self) -> List[NodeIdx]:
        return self.children

    def build(self) -> None:
        if self.built:
            return
        self.build_inner()
        self.built = True

    @abstractmethod
    def build_inner(self) -> None:
        pass


class DecNodeParent(INodeParent, DecNode):
    def __init__(
        self,
        idx: NodeIdx,
        alg_root: IAlgRoot,
        log_level: int = logging.INFO,
        **kwargs,
    ) -> None:
        self.alg_root = alg_root
        self.log_level = log_level
        super().__init__(idx, **kwargs)

    def get_alg_root(self) -> IAlgRoot:
        return self.alg_root

    def add_child(self, idx: NodeIdx, multiplier: float = 1.0) -> None:
        if idx in self.children:
            raise ValueError(f"Idx {idx} already in children of node {self.idx}")
        self.children.append(idx)
        self.set_multiplier(idx, multiplier)

    def set_groups(self, groups: List[List[NodeIdx]]) -> None:
        all_group_elements = set()
        for group in groups:
            group_set = set(group)
            if not all_group_elements.isdisjoint(group_set):
                raise ValueError("Groups are not disjoint")
            all_group_elements.update(group_set)

        if all_group_elements != set(self.children):
            raise ValueError("Not all elements in groups coincide with children")

        self.groups = groups

    def get_groups(self) -> List[List[NodeIdx]]:
        return self.groups

    def set_multiplier(self, idx: NodeIdx, multiplier: float) -> None:
        self.children_multipliers[idx] = multiplier

    def get_multiplier(self, idx: NodeIdx) -> float:
        return self.children_multipliers[idx]

    def set_child_bound(self, idx: NodeIdx, bound: float) -> None:
        self.children_bounds[idx] = bound

    def get_child_bound(self, idx: NodeIdx) -> float:
        return self.children_bounds[idx]

    def build_inner(self) -> None:
        if len(self.groups) == 0:
            self.groups = [[child] for child in self.children]
        self.cut_aggregator = CutAggregator(self.groups, self.children_multipliers)
        self.num_cuts = len(self.groups)

        subobj_bounds: List[float | None] = []
        for group in self.groups:
            bound = 0.0
            for member in group:
                if member not in self.children_bounds:
                    bound = None
                    break
                bound += (
                    self.children_multipliers[member] * self.children_bounds[member]
                )
            subobj_bounds.append(bound)
        self.alg_root.build(subobj_bounds)

    def reset(self) -> None:
        self.alg_root.reset_iteration()

    def set_logger(self) -> None:
        assert self.depth is not None
        self.alg_root.set_logger(self.idx, self.depth, self.log_level)

    def run_step(
        self, up_messages: Dict[NodeIdx, UpMessage] | None
    ) -> Tuple[int, DnMessage]:
        if up_messages is None:
            return self.alg_root.run_step(None)
        aggregate_cuts = self.cut_aggregator.get_aggregate_cuts(up_messages)
        return self.alg_root.run_step(aggregate_cuts)

    def get_init_dn_message(self, **kwargs) -> InitDnMessage:
        init_message = self.alg_root.get_init_dn_message(**kwargs)
        init_message.set_depth(self.get_depth())
        return init_message

    def pass_init_up_messages(self, messages: Dict[NodeIdx, InitUpMessage]) -> None:
        for node_id, message in messages.items():
            bound = message.get_bound()
            if bound is None:
                continue
            self.set_child_bound(node_id, bound)

    def get_final_dn_message(self, **kwargs) -> FinalDnMessage:
        return self.alg_root.get_final_dn_message(**kwargs)

    def pass_final_up_message(
        self, messages: Dict[NodeIdx, FinalUpMessage]
    ) -> FinalUpMessage:
        message_list: list[FinalUpMessage] = []
        multiplier_list: list[float] = []
        for node_id, message in messages.items():
            message_list.append(message)
            multiplier_list.append(self.get_multiplier(node_id))
        return self.alg_root.pass_final_up_message(message_list, multiplier_list)

    def get_num_vars(self) -> int:
        return self.alg_root.get_num_vars()

    def add_cuts(self, up_messages: Dict[int, UpMessage]) -> None:
        aggregate_cuts = self.cut_aggregator.get_aggregate_cuts(up_messages)
        self.alg_root.add_cuts(aggregate_cuts)

    def save(self, dir: Path) -> None:
        node_dir = dir / f"node{self.idx}"
        create_directory(node_dir)
        self.alg_root.save(node_dir)


DecNodeRoot = DecNodeParent


class DecNodeChild(INodeChild, DecNode):
    def __init__(self, idx: NodeIdx, alg_leaf: IAlgLeaf, **kwargs) -> None:
        self.alg_leaf = alg_leaf
        self.bound = None
        super().__init__(idx, **kwargs)

    def get_alg_leaf(self) -> IAlgLeaf:
        return self.alg_leaf

    def add_parent(self, idx: NodeIdx) -> None:
        if idx in self.parents:
            raise ValueError(f"Idx {idx} already in parents of node {self.idx}")
        self.parents.append(idx)

    def set_bound(self, bound: float) -> None:
        self.bound = bound

    def get_bound(self) -> float | None:
        return self.bound

    def build_inner(self) -> None:
        self.alg_leaf.build()

    def pass_init_dn_message(self, message: InitDnMessage) -> None:
        self.set_depth(message.get_depth() + 1)
        self.alg_leaf.pass_init_dn_message(message)

    def get_init_up_message(self) -> InitUpMessage:
        message = self.alg_leaf.get_init_up_message()
        message.set_bound(self.bound)
        return message

    def pass_dn_message(self, message: DnMessage) -> None:
        self.alg_leaf.pass_dn_message(message)

    def get_up_message(self) -> UpMessage:
        return self.alg_leaf.get_up_message()

    def pass_final_dn_message(self, message: FinalDnMessage) -> None:
        return self.alg_leaf.pass_final_dn_message(message)

    def get_final_up_message(self) -> FinalUpMessage:
        return self.alg_leaf.get_final_up_message()

    def solve(self, message: DnMessage) -> UpMessage:
        self.pass_dn_message(message)
        return self.get_up_message()

    def save(self, dir: Path) -> None:
        node_dir = dir / f"node{self.idx}"
        create_directory(node_dir)
        self.alg_leaf.save(node_dir)


DecNodeLeaf = DecNodeChild


class DecNodeInner(INodeInner, DecNodeParent, DecNodeChild):
    def __init__(
        self,
        idx: NodeIdx,
        alg_root: IAlgRoot,
        alg_leaf: IAlgLeaf,
        log_level: int = logging.INFO,
    ) -> None:
        super().__init__(
            idx=idx, alg_root=alg_root, alg_leaf=alg_leaf, log_level=log_level
        )

    def build_inner(self) -> None:
        DecNodeParent.build_inner(self)
        DecNodeChild.build_inner(self)

    def save(self, dir: Path):
        DecNodeParent.save(self, dir)

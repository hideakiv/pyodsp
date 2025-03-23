from abc import ABC, abstractmethod
from typing import Dict, List

from ._node import NodeIdx, INode, INodeParent, INodeChild, INodeInner

class DecNode(INode, ABC):
    def __init__(self, idx: NodeIdx) -> None:
        self.idx: NodeIdx = idx
        self.depth: int | None = None

    def get_idx(self) -> NodeIdx:
        return self.idx

    def get_depth(self) -> int:
        if self.depth is None:
            raise ValueError("Depth not initialized")
        return self.depth

    def set_depth(self, depth: int) -> None:
        self.depth = depth

class DecNodeParent(INodeParent, DecNode, ABC):
    def __init__(self, idx: NodeIdx) -> None:
        super().__init__(idx)
        self.children: List[NodeIdx] = []
        self.children_multipliers: Dict[NodeIdx, float] = {}
        self.groups = []

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

    def set_multiplier(self, idx: NodeIdx, multiplier: float) -> None:
        self.children_multipliers[idx] = multiplier

    def get_multiplier(self, idx: NodeIdx) -> float:
        return self.children_multipliers[idx]
    
    def get_parents(self) -> List[NodeIdx]:
        return []

    def get_children(self) -> List[NodeIdx]:
        return self.children
    
DecNodeRoot = DecNodeParent

class DecNodeChild(INodeChild, DecNode, ABC):
    def __init__(self, idx: NodeIdx) -> None:
        super().__init__(idx)
        self.parents: List[NodeIdx] = []

    def add_parent(self, idx: NodeIdx) -> None:
        if idx in self.parents:
            raise ValueError(f"Idx {idx} already in parents of node {self.idx}")
        self.parents.append(idx)

    def get_parents(self) -> List[NodeIdx]:
        return self.parents
    
    def get_children(self) -> List[NodeIdx]:
        return []

DecNodeLeaf = DecNodeChild

class DecNodeInner(INodeInner, DecNodeParent, DecNodeChild, ABC):
    def __init__(self, idx: NodeIdx) -> None:
        DecNodeParent.__init__(self, idx)
        DecNodeChild.__init__(self, idx)  # slightly inefficient

    def get_parents(self) -> List[NodeIdx]:
        return DecNodeChild.get_parents(self)
    
    def get_children(self) -> List[NodeIdx]:
        return DecNodeParent.get_children(self)
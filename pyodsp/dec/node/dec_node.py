from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from pathlib import Path

from pyodsp.alg.cuts import Cut

from ._node import NodeIdx, INode, INodeParent, INodeChild, INodeInner
from ._alg import IAlgRoot, IAlgLeaf
from .cut_aggregator import CutAggregator
from ..run._message import IMessage
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

        self.kwargs = kwargs

        self.built = False

    def get_kwargs(self) -> Dict[str, Any]:
        return self.kwargs

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
    def __init__(self, idx: NodeIdx, alg_root: IAlgRoot, **kwargs) -> None:
        self.alg_root = alg_root
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
        self.alg_root.set_logger(self.idx, self.depth)

    def run_step(self, cuts: Dict[int, Cut] | None) -> Tuple[int, List[float]]:
        if cuts is None:
            return self.alg_root.run_step(None)
        aggregate_cuts = self.cut_aggregator.get_aggregate_cuts(cuts)
        return self.alg_root.run_step(aggregate_cuts)

    def get_init_message(self, **kwargs) -> IMessage:
        return self.alg_root.get_init_message(**kwargs)

    def get_solution_dn(self) -> List[float]:
        return self.alg_root.get_solution_dn()

    def get_num_vars(self) -> int:
        return self.alg_root.get_num_vars()

    def add_cuts(self, cuts: Dict[int, Cut]) -> None:
        aggregate_cuts = self.cut_aggregator.get_aggregate_cuts(cuts)
        self.alg_root.add_cuts(aggregate_cuts)

    def save(self, dir: Path) -> None:
        node_dir = dir / f"node{self.idx}"
        create_directory(node_dir)
        self.alg_root.save(node_dir)

    def is_minimize(self) -> bool:
        return self.alg_root.is_minimize()


DecNodeRoot = DecNodeParent


class DecNodeChild(INodeChild, DecNode):
    def __init__(self, idx: NodeIdx, alg_leaf: IAlgLeaf, **kwargs) -> None:
        self.alg_leaf = alg_leaf
        super().__init__(idx, **kwargs)

    def get_alg_leaf(self) -> IAlgLeaf:
        return self.alg_leaf

    def add_parent(self, idx: NodeIdx) -> None:
        if idx in self.parents:
            raise ValueError(f"Idx {idx} already in parents of node {self.idx}")
        self.parents.append(idx)

    def set_bound(self, bound: float) -> None:
        self.bound = bound

    def get_bound(self) -> float:
        return self.bound

    def get_objective_value(self) -> float:
        return self.alg_leaf.get_objective_value()

    def build_inner(self) -> None:
        self.alg_leaf.build()

    def pass_init_message(self, message: IMessage) -> None:
        self.alg_leaf.pass_init_message(message)

    def pass_solution(self, solution: List[float]) -> None:
        self.alg_leaf.pass_solution(solution)

    def pass_final_message(self, message: IMessage) -> None:
        return self.alg_leaf.pass_final_message(message)

    def get_subgradient(self) -> Cut:
        return self.alg_leaf.get_subgradient()

    def solve(self, solution: List[float]) -> Cut:
        self.pass_solution(solution)
        return self.get_subgradient()

    def save(self, dir: Path) -> None:
        node_dir = dir / f"node{self.idx}"
        create_directory(node_dir)
        self.alg_leaf.save(node_dir)

    def is_minimize(self) -> bool:
        return self.alg_leaf.is_minimize()


DecNodeLeaf = DecNodeChild


class DecNodeInner(INodeInner, DecNodeParent, DecNodeChild):
    def __init__(self, idx: NodeIdx, alg_root: IAlgRoot, alg_leaf: IAlgLeaf) -> None:
        super().__init__(idx=idx, alg_root=alg_root, alg_leaf=alg_leaf)

    def build_inner(self) -> None:
        DecNodeParent.build_inner(self)
        DecNodeChild.build_inner(self)

    def save(self, dir: Path):
        DecNodeParent.save(self, dir)

    def is_minimize(self) -> bool:
        return DecNodeParent.is_minimize(self)

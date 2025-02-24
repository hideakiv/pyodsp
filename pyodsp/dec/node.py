from typing import List, Dict


class Node:
    def __init__(
        self,
        idx: int,
        parent: int | None = None,
    ) -> None:
        self.idx: int = idx
        self.parent: int = parent
        self.children: List[int] = []
        self.children_multipliers: Dict[int, float] = {}
        self.depth = None

    def add_child(self, idx: int, multiplier: float = 1.0):
        self.children.append(idx)
        self.children_multipliers[idx] = multiplier

    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent

    def remove_child(self, idx: int):
        if idx in self.children:
            self.children.remove(idx)
            self.children_multipliers.pop(idx)

    def remove_children(self):
        self.children = []
        self.children_multipliers = {}

    def set_parent(self, parent):
        self.parent = parent

    def get_depth(self) -> int:
        return self.depth

    def set_depth(self, depth: int) -> None:
        self.depth = depth
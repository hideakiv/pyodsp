from typing import List

class Node:
    def __init__(
            self,
            idx: int,
            parent: int | None = None,
        ) -> None:
        self.idx: int = idx
        self.parent: int = parent
        self.children: List[int] = []

    def add_child(self, idx: int):
        self.children.append(idx)

    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent

    def remove_child(self, idx: int):
        if idx in self.children:
            self.children.remove(idx)

    def remove_children(self):
        self.children = []

    def set_parent(self, parent):
        self.parent = parent


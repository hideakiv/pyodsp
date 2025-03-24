from typing import List, Dict

from ..node.dec_node import DecNodeLeaf
from .alg_leaf import DdAlgLeaf


class DdLeafNode(DecNodeLeaf):
    def __init__(
        self,
        idx: int,
        alg_leaf: DdAlgLeaf,
        parent: int,
    ) -> None:
        super().__init__(idx, alg_leaf)
        self.add_parent(parent)

    def set_coupling_matrix(self, coupling_matrix: List[Dict[int, float]]) -> None:
        self.alg_leaf.set_coupling_matrix(coupling_matrix)
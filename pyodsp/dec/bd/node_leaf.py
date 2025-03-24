
from ..node.dec_node import DecNodeLeaf
from .alg_leaf import BdAlgLeaf


class BdLeafNode(DecNodeLeaf):
    def __init__(
        self,
        idx: int,
        alg_leaf: BdAlgLeaf,
        bound: float,
        parent: int,
    ) -> None:
        super().__init__(idx, alg_leaf)
        self.add_parent(parent)
        self.bound = bound

    def get_bound(self) -> float:
        return self.bound

from typing import List

from pyodsp.alg.cuts import Cut

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

    def solve(self, coupling_values: List[float]) -> Cut:
        self.alg_leaf.pass_solution(coupling_values)
        return self.alg_leaf.get_subgradient()

    def get_bound(self) -> float:
        return self.bound

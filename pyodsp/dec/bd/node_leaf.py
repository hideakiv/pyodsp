from typing import List
from pathlib import Path

from pyodsp.alg.cuts import Cut

from ..node.dec_node import DecNodeLeaf
from .alg_leaf import BdAlgLeaf
from ..utils import create_directory


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

        self.built = False

    def build(self) -> None:
        if self.built:
            return
        self.alg_leaf.build()
        self.built = True

    def solve(self, coupling_values: List[float]) -> Cut:
        self.alg_leaf.fix_variables(coupling_values)
        return self.alg_leaf.get_subgradient()

    def get_bound(self) -> float:
        return self.bound

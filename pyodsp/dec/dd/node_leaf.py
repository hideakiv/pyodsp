from typing import List, Dict

from pyodsp.alg.cuts import Cut

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
        self._is_minimize = alg_leaf.is_minimize()
        self.len_vars = alg_leaf.get_len_vars()

    def set_coupling_matrix(self, coupling_matrix: List[Dict[int, float]]) -> None:
        self.alg_leaf.set_coupling_matrix(coupling_matrix)

    def solve(self, dual_values: List[float]) -> Cut:
        self.alg_leaf.pass_solution(dual_values)
        return self.alg_leaf.get_subgradient()
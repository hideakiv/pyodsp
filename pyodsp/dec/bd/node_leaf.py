from typing import List
from pathlib import Path

from pyodsp.alg.cuts import Cut

from .node import BdNode
from .alg_leaf import BdAlgLeaf
from ..utils import create_directory


class BdLeafNode(BdNode):
    def __init__(
        self,
        idx: int,
        alg: BdAlgLeaf,
        bound: float,
        parent: int,
    ) -> None:
        super().__init__(idx, parent=parent)
        self.alg = alg
        self.bound = bound

        self.built = False

    def build(self) -> None:
        if self.built:
            return
        self.alg.build()
        self.built = True

    def solve(self, coupling_values: List[float]) -> Cut:
        self.alg.fix_variables(coupling_values)
        return self.alg.get_subgradient()

    def get_bound(self) -> float:
        return self.bound

    def save(self, dir: Path):
        node_dir = dir / f"node{self.idx}"
        create_directory(node_dir)
        self.alg.save(node_dir)

    def is_minimize(self) -> bool:
        return self.alg.is_minimize()

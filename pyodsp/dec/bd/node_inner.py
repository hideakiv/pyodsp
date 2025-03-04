from typing import List

from pyodsp.alg.cuts import Cut

from .node_root import BdRootNode
from .alg_root import BdAlgRoot
from .alg_leaf import BdAlgLeaf

class BdInnerNode(BdRootNode):
    def __init__(
        self,
        idx: int,
        alg_root: BdAlgRoot,
        alg_leaf: BdAlgLeaf,
        bound: float,
        parent: int,
    ) -> None:
        super().__init__(idx, alg_root)
        self.alg_leaf = alg_leaf
        self.bound = bound
        self.parent = parent
    
    def build(self) -> None:
        super().build()
        
        self.alg_leaf.build()

    def fix_variables(self, coupling_values: List[float]) -> None:
        self.alg_leaf.fix_variables(coupling_values)

    def get_subgradient(self) -> Cut:
        return self.alg_leaf.get_subgradient()

    def get_bound(self) -> float:
        return self.bound



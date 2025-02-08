from typing import List

from pyomo.core.base.var import VarData

from pyodec.alg.bm.cuts import Cut

from .node import BdNode
from .solver_leaf import BdAlgLeaf


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

from typing import List

from pyomo.core.base.var import VarData

from ..node.dec_node import DecNodeRoot
from .alg_root import BdAlgRoot


class BdRootNode(DecNodeRoot):

    def __init__(
        self,
        idx: int,
        alg_root: BdAlgRoot,
    ) -> None:
        super().__init__(idx, alg_root)
        self.coupling_vars_dn: List[VarData] = alg_root.get_vars()

    def get_solution_dn(self) -> List[float]:
        return [var.value for var in self.coupling_vars_dn]


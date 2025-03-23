from typing import List, Dict

from pyomo.core.base.var import VarData

from pyodsp.alg.cuts import Cut

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
    
    def add_cuts(self, cuts: Dict[int, Cut]) -> None:
        aggregate_cuts = self.cut_aggregator.get_aggregate_cuts(cuts)
        self.alg_root.add_cuts(aggregate_cuts)

    def get_solution_dn(self) -> List[float]:
        return [var.value for var in self.coupling_vars_dn]


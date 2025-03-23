from typing import List, Dict, Tuple

from pyomo.core.base.var import VarData

from pyodsp.alg.cuts import Cut

from ..node.dec_node import DecNodeRoot
from ..node.cut_aggregator import CutAggregator
from .alg_root import BdAlgRoot


class BdRootNode(DecNodeRoot):

    def __init__(
        self,
        idx: int,
        alg_root: BdAlgRoot,
    ) -> None:
        super().__init__(idx, alg_root)
        self.coupling_vars_dn: List[VarData] = alg_root.get_vars()

        self.children_bounds: Dict[int, float] = {}

        self.built = False

    def set_bound(self, idx, bound) -> None:
        self.children_bounds[idx] = bound
    
    def set_logger(self):
        self.alg_root.set_logger(self.idx, self.depth)

    def build(self) -> None:
        if self.built:
            return
        if len(self.groups) == 0:
            self.groups = [[child] for child in self.children]
        self.cut_aggregator = CutAggregator(self.groups, self.children_multipliers)
        self.num_cuts = len(self.groups)

        subobj_bounds = []
        for group in self.groups:
            bound = 0.0
            for member in group:
                bound += (
                    self.children_multipliers[member] * self.children_bounds[member]
                )
            subobj_bounds.append(bound)

        self.alg_root.build(subobj_bounds)
        self.built = True

    def run_step(self, cuts: Dict[int, Cut] | None) -> Tuple[int, List[float]]:
        if cuts is None:
            return self.alg_root.run_step(None)
        aggregate_cuts = self.cut_aggregator.get_aggregate_cuts(cuts)
        return self.alg_root.run_step(aggregate_cuts)
    
    def add_cuts(self, cuts: Dict[int, Cut]) -> None:
        aggregate_cuts = self.cut_aggregator.get_aggregate_cuts(cuts)
        self.alg_root.add_cuts(aggregate_cuts)

    def get_solution_dn(self) -> List[float]:
        return [var.value for var in self.coupling_vars_dn]


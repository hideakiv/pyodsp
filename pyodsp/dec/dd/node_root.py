from typing import List, Dict, Tuple

from pyomo.core.base.var import VarData

from pyodsp.alg.cuts import Cut

from ..node.dec_node import DecNodeRoot
from ..node.cut_aggregator import CutAggregator
from .alg_root import DdAlgRoot
from .mip_heuristic_root import MipHeuristicRoot


class DdRootNode(DecNodeRoot):

    def __init__(self, idx: int, alg_root: DdAlgRoot, rootsolver: str, **kwargs) -> None:
        super().__init__(idx, alg_root)
        self.coupling_vars_dn: Dict[int, List[VarData]] = alg_root.get_vars_dn()
        self.num_constrs = alg_root.num_constrs

        self.built = False

        self.rootsolver = rootsolver
        self.kwargs = kwargs

    def build(self) -> None:
        if self.built:
            return
        if len(self.groups) == 0:
            self.groups = [[child] for child in self.children]
        self.cut_aggregator = CutAggregator(self.groups, self.children_multipliers)
        self.num_cuts = len(self.groups)

        self.alg_root.build(self.num_cuts)
        self.built = True

    def solve_mip_heuristic(self) -> Dict[int, List[float]]:
        self.mip_heuristic = MipHeuristicRoot(
            self.rootsolver, self.groups, self.alg_root, **self.kwargs
        )
        self.mip_heuristic.build()
        return self.mip_heuristic.run()

    def run_step(self, cuts: Dict[int, Cut] | None) -> Tuple[int, List[float]]:
        if cuts is None:
            return self.alg_root.run_step(None)
        aggregate_cuts = self.cut_aggregator.get_aggregate_cuts(cuts)
        return self.alg_root.run_step(aggregate_cuts)

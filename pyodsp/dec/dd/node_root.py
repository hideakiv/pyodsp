from typing import List, Dict

from pyomo.core.base.var import VarData

from ..node.dec_node import DecNodeRoot
from .alg_root import DdAlgRoot
from .mip_heuristic_root import MipHeuristicRoot


class DdRootNode(DecNodeRoot):

    def __init__(self, idx: int, alg_root: DdAlgRoot, rootsolver: str, **kwargs) -> None:
        super().__init__(idx, alg_root)

        self.rootsolver = rootsolver
        self.kwargs = kwargs

    def solve_mip_heuristic(self) -> Dict[int, List[float]]:
        self.mip_heuristic = MipHeuristicRoot(
            self.rootsolver, self.groups, self.alg_root, **self.kwargs
        )
        self.mip_heuristic.build()
        return self.mip_heuristic.run()

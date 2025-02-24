from typing import List, Dict
from pathlib import Path

from pyomo.core.base.var import VarData

from pyodsp.alg.cuts import Cut, CutList

from .node import DdNode
from .alg_root import DdAlgRoot
from .mip_heuristic_root import MipHeuristicRoot
from ..utils import create_directory


class DdRootNode(DdNode):

    def __init__(self, idx: int, alg: DdAlgRoot, rootsolver: str, **kwargs) -> None:
        super().__init__(idx, parent=None)
        self.alg = alg
        self.coupling_vars_dn: Dict[int, List[VarData]] = alg.get_vars_dn()
        self.is_minimize = alg.is_minimize
        self.num_constrs = alg.num_constrs

        self.groups = None
        self.built = False

        self.rootsolver = rootsolver
        self.kwargs = kwargs

    def set_groups(self, groups: List[List[int]]):
        # self.groups = groups
        raise ValueError("No support for set_groups")  # TODO?

    def add_child(self, idx: int, multiplier: float = 1.0):
        if multiplier != 1.0:
            raise ValueError("No support for multipliers in dd")
        super().add_child(idx, multiplier)

    def build(self) -> None:
        if self.built:
            return
        if self.groups is None:
            self.groups = [[child] for child in self.children]
        self.num_cuts = len(self.groups)

        self.alg.build(self.num_cuts)
        self.built = True

    def solve_mip_heuristic(self) -> Dict[int, List[float]]:
        self.mip_heuristic = MipHeuristicRoot(
            self.rootsolver, self.groups, self.alg, **self.kwargs
        )
        self.mip_heuristic.build()
        return self.mip_heuristic.run()

    def run_step(self, cuts: Dict[int, Cut] | None) -> List[float] | None:
        if cuts is None:
            return self.alg.run_step(None)
        aggregate_cuts = []
        assert self.groups is not None
        for group in self.groups:
            assert len(group) == 1
            child = group[0]
            aggregate_cuts.append(CutList([cuts[child]]))
        return self.alg.run_step(aggregate_cuts)

    def save(self, dir: Path):
        node_dir = dir / f"node{self.idx}"
        create_directory(node_dir)
        self.alg.save(node_dir)

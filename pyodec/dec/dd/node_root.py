from multiprocessing import Value
from typing import List, Dict
from pathlib import Path

from pyomo.core.base.var import VarData

from pyodec.alg.cuts import Cut, OptimalityCut, FeasibilityCut, CutList

from .node import DdNode
from .alg_root import DdAlgRoot
from ..utils import create_directory


class DdRootNode(DdNode):

    def __init__(self, idx: int, alg: DdAlgRoot) -> None:
        super().__init__(idx, parent=None)
        self.alg = alg
        self.coupling_vars_dn: Dict[int, List[VarData]] = alg.get_vars_dn()
        self.is_minimize = alg.is_minimize
        self.num_constrs = alg.num_constrs

        self.groups = None
        self.built = False

    def set_groups(self, groups: List[List[int]]):
        # self.groups = groups
        raise ValueError("No support for set_groups yet")

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

    def run_step(self, cuts: Dict[int, Cut] | None) -> List[float] | None:
        if cuts is None:
            return self.alg.run_step(None)
        aggregate_cuts = []
        assert self.groups is not None
        for group in self.groups:
            group_cut = []
            group_multipliers = []
            for child in group:
                group_cut.append(cuts[child])
                group_multipliers.append(self.children_multipliers[child])
            aggregate_cut = self._aggregate_cuts(group_multipliers, group_cut)

            aggregate_cuts.append(aggregate_cut)
        return self.alg.run_step(aggregate_cuts)

    def save(self, dir: Path):
        node_dir = dir / f"node{self.idx}"
        create_directory(node_dir)
        self.alg.save(node_dir)

    def _aggregate_cuts(self, multipliers: List[float], cuts: List[Cut]) -> CutList:
        new_coef = [0.0 for _ in range(self.num_constrs)]
        new_constant = 0
        new_objective = 0
        feasibility_cuts = []
        for multiplier, cut in zip(multipliers, cuts):
            if isinstance(cut, OptimalityCut):
                if len(feasibility_cuts) == 0:
                    new_coef = [
                        new_coef[i] + multiplier * cut.coeffs[i]
                        for i in range(self.num_constrs)
                    ]
                    new_constant += multiplier * cut.rhs
                    new_objective += multiplier * cut.objective_value
            elif isinstance(cut, FeasibilityCut):
                feasibility_cuts.append(cut)
        if len(feasibility_cuts) > 0:
            return CutList(feasibility_cuts)
        return CutList(
            [
                OptimalityCut(
                    coeffs=new_coef,
                    rhs=new_constant,
                    objective_value=new_objective,
                )
            ]
        )

from typing import List, Dict
from pathlib import Path

from pyomo.core.base.var import VarData

from pyodsp.alg.cuts import Cut, OptimalityCut, FeasibilityCut, CutList
from pyodsp.alg.const import DEC_CUT_ABS_TOL

from .node import BdNode
from .alg_root import BdAlgRoot
from ..utils import create_directory


class BdRootNode(BdNode):

    def __init__(
        self,
        idx: int,
        alg: BdAlgRoot,
    ) -> None:
        super().__init__(idx, parent=None)
        self.alg = alg
        self.coupling_vars_dn: List[VarData] = alg.get_vars()

        self.children_bounds: Dict[int, float] = {}

        self.groups = None
        self.built = False

    def set_bound(self, idx, bound) -> None:
        self.children_bounds[idx] = bound

    def set_groups(self, groups: List[List[int]]):
        self.groups = groups
    
    def set_logger(self):
        self.alg.set_logger(self.idx, self.depth)

    def remove_child(self, idx):
        self.children_bounds.pop(idx)
        return super().remove_child(idx)

    def remove_children(self):
        self.children_bounds = {}
        return super().remove_children()

    def build(self) -> None:
        if self.built:
            return
        if self.groups is None:
            self.groups = [[child] for child in self.children]
        self.num_cuts = len(self.groups)

        subobj_bounds = []
        for group in self.groups:
            bound = 0.0
            for member in group:
                bound += (
                    self.children_multipliers[member] * self.children_bounds[member]
                )
            subobj_bounds.append(bound)

        self.alg.build(subobj_bounds)
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

    def is_minimize(self) -> bool:
        return self.alg.is_minimize()

    def _aggregate_cuts(self, multipliers: List[float], cuts: List[Cut]) -> CutList:
        new_coef = [0.0 for _ in range(len(self.coupling_vars_dn))]
        new_constant = 0
        new_objective = 0
        feasibility_cuts = []
        for multiplier, cut in zip(multipliers, cuts):
            if isinstance(cut, OptimalityCut):
                if len(feasibility_cuts) == 0:
                    for i, coeff in cut.coeffs.items():
                        new_coef[i] += multiplier * coeff
                    new_constant += multiplier * cut.rhs
                    new_objective += multiplier * cut.objective_value
            elif isinstance(cut, FeasibilityCut):
                feasibility_cuts.append(cut)
        if len(feasibility_cuts) > 0:
            return CutList(feasibility_cuts)
        sparse_coeff = {j: val for j, val in enumerate(new_coef) if abs(val) > DEC_CUT_ABS_TOL}
        return CutList(
            [
                OptimalityCut(
                    coeffs=sparse_coeff,
                    rhs=new_constant,
                    objective_value=new_objective,
                    info={},
                )
            ]
        )

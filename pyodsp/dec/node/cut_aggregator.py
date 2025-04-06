from typing import List, Dict

from pyodsp.alg.cuts import Cut, OptimalityCut, FeasibilityCut, CutList
from ._node import NodeIdx


class CutAggregator:
    def __init__(
        self, groups: List[List[NodeIdx]], multipliers: Dict[NodeIdx, float]
    ) -> None:
        self.groups = groups
        self.group_multipliers = [
            [multipliers[idx] for idx in group] for group in groups
        ]

    def get_aggregate_cuts(self, cuts: Dict[int, Cut]) -> List[CutList]:
        aggregate_cuts = []
        for i, group in enumerate(self.groups):
            group_cut = []
            for child in group:
                group_cut.append(cuts[child])
            aggregate_cut = self._get_aggregate_cut(
                self.group_multipliers[i], group_cut
            )

            aggregate_cuts.append(aggregate_cut)
        return aggregate_cuts

    def _get_aggregate_cut(self, multipliers: List[float], cuts: List[Cut]) -> CutList:
        new_coeff = {}
        new_constant = 0
        new_objective = 0
        new_info = {}
        feasibility_cuts = []
        for multiplier, cut in zip(multipliers, cuts):
            if isinstance(cut, OptimalityCut):
                if len(feasibility_cuts) == 0:
                    for i, coeff in cut.coeffs.items():
                        new_coeff[i] = new_coeff.get(i, 0) + multiplier * coeff
                    new_constant += multiplier * cut.rhs
                    new_objective += multiplier * cut.objective_value
                    new_info = cut.info
            elif isinstance(cut, FeasibilityCut):
                feasibility_cuts.append(cut)
        if len(feasibility_cuts) > 0:
            return CutList(feasibility_cuts)
        return CutList(
            [
                OptimalityCut(
                    coeffs=new_coeff,
                    rhs=new_constant,
                    objective_value=new_objective,
                    info=new_info,
                )
            ]
        )

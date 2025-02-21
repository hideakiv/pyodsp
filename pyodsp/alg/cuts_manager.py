from typing import List
from dataclasses import dataclass

from pyomo.environ import Constraint

from .cuts import Cut, FeasibilityCut, OptimalityCut
from .const import BM_SLACK_TOLERANCE, BM_MAX_CUT_AGE, BM_CUT_SIM_TOLERANCE


@dataclass
class CutInfo:
    """Class for cut with info"""

    constraint: Constraint
    cut: Cut
    idx: int
    iteration: int
    trial_point: List[float]
    age: int


class CutsManager:
    def __init__(self) -> None:
        self._active_cuts: List[List[CutInfo]] = []

        self._num_optimality: List[int] = []
        self._num_feasibility: List[int] = []

    def build(self, num_idx: int) -> None:
        for _ in range(num_idx):
            self._active_cuts.append([])
            self._num_optimality.append(0)
            self._num_feasibility.append(0)

    def get_num_optimality(self, idx: int) -> int:
        return self._num_optimality[idx]

    def get_num_feasibility(self, idx: int) -> int:
        return self._num_feasibility[idx]

    def append_cut(self, cut_info: CutInfo) -> None:
        idx = cut_info.idx
        if isinstance(cut_info.cut, OptimalityCut):
            self._num_optimality[idx] += 1
        elif isinstance(cut_info.cut, FeasibilityCut):
            self._num_feasibility[idx] += 1
        else:
            ValueError("Invalid cut type")
        if self._is_similar(cut_info):
            cut_info.constraint.deactivate()
        else:
            self._active_cuts[idx].append(cut_info)

    def _is_similar(self, cut_info: CutInfo) -> bool:
        for cut in self._active_cuts[cut_info.idx]:
            square = (cut_info.cut.rhs - cut.cut.rhs) ** 2
            for x, y in zip(cut_info.cut.coeffs, cut.cut.coeffs):
                square += (x - y) ** 2
            if square < BM_CUT_SIM_TOLERANCE:
                return True
        return False

    def increment(self) -> None:
        for cuts in self._active_cuts:
            for cut in cuts:
                try:
                    lslack = cut.constraint.lslack()
                    uslack = cut.constraint.uslack()
                    if lslack > BM_SLACK_TOLERANCE and uslack > BM_SLACK_TOLERANCE:
                        cut.age += 1
                    else:
                        cut.age = 0
                except Exception:
                    pass
    
    def purge(self) -> None:
        def below_max(cut: CutInfo) -> bool:
            below = cut.age < BM_MAX_CUT_AGE
            if not below:
                cut.constraint.deactivate()
            return below
        
        for cuts in self._active_cuts:
            cuts[:] = [cut for cut in cuts if below_max(cut)]

    def get_cuts(self) -> List[List[CutInfo]]:
        return self._active_cuts

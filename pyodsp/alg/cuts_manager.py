from typing import List
from dataclasses import dataclass

from pyomo.environ import Constraint

from .cuts import Cut, FeasibilityCut, OptimalityCut


@dataclass
class CutInfo:
    """Class for cut with info"""

    constraint: Constraint
    cut: Cut
    idx: int
    iteration: int
    trial_point: List[float]
    period: int


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

    def append_cut(self, cut_info) -> None:
        idx = cut_info.idx
        if isinstance(cut_info.cut, OptimalityCut):
            self._num_optimality[idx] += 1
        elif isinstance(cut_info.cut, FeasibilityCut):
            self._num_feasibility[idx] += 1
        else:
            ValueError("Invalid cut type")
        self._active_cuts[idx].append(cut_info)

    def increment(self) -> None:
        pass

    def get_cuts(self) -> List[List[CutInfo]]:
        return self._active_cuts

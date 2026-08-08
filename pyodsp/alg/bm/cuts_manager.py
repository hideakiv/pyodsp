from typing import List
from dataclasses import dataclass

from pyomo.environ import ConcreteModel, Constraint

from .cuts import Cut, FeasibilityCut, OptimalityCut
from ..params import BM_SLACK_TOLERANCE, BM_MAX_CUT_AGE, BM_CUT_SIM_TOLERANCE


@dataclass
class CutInfo:
    """Class for cut with info"""

    constraint: Constraint
    cut: Cut
    idx: int
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
            raise ValueError("Invalid cut type")
        if self._is_similar(cut_info):
            cut_info.constraint.deactivate()
        else:
            self._active_cuts[idx].append(cut_info)

    def _is_similar(self, cut_info: CutInfo) -> bool:
        for cut in self._active_cuts[cut_info.idx]:
            diff = cut_info.cut.coeffs.copy()
            for j, val in cut.cut.coeffs.items():
                if j in diff.keys():
                    diff[j] -= val
                else:
                    diff[j] = -val
            square = (cut_info.cut.rhs - cut.cut.rhs) ** 2
            for val in diff.values():
                square += val**2
            if square < BM_CUT_SIM_TOLERANCE:
                return True
        return False

    def increment(self) -> None:
        for cuts in self._active_cuts:
            if len(cuts) > 1:
                for cut in cuts:
                    lslack = cut.constraint.lslack()
                    uslack = cut.constraint.uslack()
                    if lslack > BM_SLACK_TOLERANCE and uslack > BM_SLACK_TOLERANCE:
                        cut.age += 1
                    else:
                        cut.age = 0

    def purge(self, model: ConcreteModel) -> None:
        """Remove cuts older than BM_MAX_CUT_AGE."""
        to_eliminate = [
            cut.constraint.name
            for cuts in self._active_cuts
            for cut in cuts
            if cut.age >= BM_MAX_CUT_AGE
        ]
        self.eliminate_cuts(model, to_eliminate)

    def eliminate_cuts(self, model: ConcreteModel, names: List[str]) -> None:
        """Deactivate and remove specific cuts by constraint name.

        Used by purge() (age-based) and by BundleMethod.replace_cuts to
        clear a replica's cuts wholesale before mirroring another
        BundleMethod's snapshot.
        """
        names = set(names)
        if not names:
            return
        for cuts in self._active_cuts:
            for cut in cuts:
                if cut.constraint.name in names:
                    cut.constraint.deactivate()
                    model.del_component(cut.constraint.name)
            cuts[:] = [cut for cut in cuts if cut.constraint.name not in names]

    def get_cuts(self) -> List[List[CutInfo]]:
        return self._active_cuts

    def get_num_cuts(self) -> int:
        return sum(len(cut_list) for cut_list in self._active_cuts)


class NonPurgingCutsManager(CutsManager):
    """A CutsManager that never purges on its own — no age tracking, no
    automatic removal — but still purges when explicitly instructed via
    eliminate_cuts (inherited unchanged).

    Intended for synchronized replicas (e.g. parallel SDDP Monte Carlo
    simulation workers) that must keep a cut list identical to a master
    CutsManager's without independently deciding what's stale — see
    LatticeMpi, which mirrors the master's cuts wholesale each sync round
    via BundleMethod.replace_cuts.
    """

    def increment(self) -> None:
        pass

    def purge(self, model: ConcreteModel) -> None:
        pass

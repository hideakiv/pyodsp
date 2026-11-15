from typing import List
from ..node._message import (
    InitDnMessage,
    InitUpMessage,
    FinalDnMessage,
    FinalUpMessage,
    DnMessage,
    UpMessage,
)
from pyodsp.alg.bm.cuts import Cut, CutList


class BdScInitDnMessage(InitDnMessage):
    def __init__(self) -> None:
        pass

    def set_depth(self, depth: int) -> None:
        self.depth = depth

    def get_depth(self) -> int:
        return self.depth


class BdScInitUpMessage(InitUpMessage):
    def __init__(self) -> None:
        self.bound = None
        self.sense_multiplier = 1.0

    def set_bound(self, bound: float | None) -> None:
        self.bound = bound

    def get_bound(self) -> float | None:
        return self.bound

    def set_sense_multiplier(self, multiplier: float) -> None:
        self.sense_multiplier = multiplier

    def get_sense_multiplier(self) -> float:
        return self.sense_multiplier


class _Origin:
    """Which master a dn message's trial point came from, carried back up
    on the up messages priced against it.

    Only sddpsc sets this: in a lattice several masters at a stage share
    one set of children, and a child's reply (its c, and the scaled cut's
    validity) is relative to the master whose rho and cuts it was priced
    against -- see SddpScAlgRootBm.add_cuts. bdsc's single master leaves
    it unset.
    """

    origin = None
    origin_multipliers: dict | None = None

    def set_origin(self, origin, multipliers: dict | None) -> None:
        self.origin = origin
        self.origin_multipliers = multipliers

    def get_origin(self):
        return self.origin

    def get_origin_multipliers(self) -> dict | None:
        return self.origin_multipliers


class BdScUpMessage(_Origin, UpMessage):
    def __init__(self, cut: Cut, c: float, tau: float, objective: float) -> None:
        self.cut = cut
        self.c = c
        self.tau = tau
        self.objective = objective

    def get_cut(self):
        return self.cut

    def get_objective(self):
        return self.objective

    def get_c(self):
        return self.c

    def get_tau(self):
        return self.tau


class BdScDnMessage(_Origin, DnMessage):
    def __init__(
        self,
        solution: List[float],
        rho: float,
        cut_list: list[CutList] | None,
        subobj_bounds: list[float],
        objective: float = 0.0,
    ) -> None:
        self.solution = solution
        self.rho = rho
        self.cut_list = cut_list
        self.subobj_bounds = subobj_bounds
        self.objective = objective

    def get_solution(self):
        return self.solution

    def get_rho(self):
        return self.rho

    def get_cut_list(self) -> list[CutList] | None:
        """The master's complete current cut set, or None if it is
        unchanged since the last message. Recipients mirror it wholesale
        rather than appending — see BdScAlgRootBm.run_step for why the
        increment alone is not enough.
        """
        return self.cut_list

    def get_subobj_bounds(self) -> list[float]:
        return self.subobj_bounds

    def get_objective(self):
        return self.objective


class BdScFinalDnMessage(FinalDnMessage):
    def __init__(self, solution: List[float] | None) -> None:
        self.solution = solution

    def get_solution(self) -> List[float] | None:
        return self.solution


class BdScFinalUpMessage(FinalUpMessage):
    def __init__(self, objective: float | None) -> None:
        self.objective = objective

    def set_objective(self, obj: float) -> None:
        self.objective = obj

    def get_objective(self) -> float | None:
        return self.objective

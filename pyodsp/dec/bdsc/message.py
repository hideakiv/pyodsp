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
    def __init__(self, is_minimize: bool) -> None:
        self.is_minimize = is_minimize

    def get_is_minimize(self) -> bool:
        return self.is_minimize

    def set_depth(self, depth: int) -> None:
        self.depth = depth

    def get_depth(self) -> int:
        return self.depth


class BdScInitUpMessage(InitUpMessage):
    def __init__(self) -> None:
        self.bound = None

    def set_bound(self, bound: float | None) -> None:
        self.bound = bound

    def get_bound(self) -> float | None:
        return self.bound


class BdScUpMessage(UpMessage):
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


class BdScDnMessage(DnMessage):
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

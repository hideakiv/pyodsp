from typing import List
from ..node._message import (
    InitDnMessage,
    InitUpMessage,
    FinalDnMessage,
    FinalUpMessage,
    DnMessage,
    UpMessage,
)
from pyodsp.alg.bm.cuts import Cut


class BdInitDnMessage(InitDnMessage):
    def __init__(self, is_minimize: bool) -> None:
        self.is_minimize = is_minimize

    def get_is_minimize(self) -> bool:
        return self.is_minimize

    def set_depth(self, depth: int) -> None:
        self.depth = depth

    def get_depth(self) -> int:
        return self.depth


class BdInitUpMessage(InitUpMessage):
    def __init__(self) -> None:
        self.bound = None

    def set_bound(self, bound: float | None) -> None:
        self.bound = bound

    def get_bound(self) -> float | None:
        return self.bound


class BdUpMessage(UpMessage):
    def __init__(self, cut: Cut) -> None:
        self.cut = cut

    def get_cut(self):
        return self.cut


class BdDnMessage(DnMessage):
    def __init__(self, solution: List[float]) -> None:
        self.solution = solution

    def get_solution(self):
        return self.solution


class BdFinalDnMessage(FinalDnMessage):
    def __init__(self, solution: List[float] | None) -> None:
        self.solution = solution

    def get_solution(self) -> List[float] | None:
        return self.solution


class BdFinalUpMessage(FinalUpMessage):
    def __init__(self, objective: float | None) -> None:
        self.objective = objective

    def get_objective(self) -> float | None:
        return self.objective

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
    def __init__(self) -> None:
        pass

    def set_depth(self, depth: int) -> None:
        self.depth = depth

    def get_depth(self) -> int:
        return self.depth


class BdInitUpMessage(InitUpMessage):
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


class BdUpMessage(UpMessage):
    def __init__(self, cut: Cut, objective: float) -> None:
        self.cut = cut
        self.objective = objective

    def get_cut(self):
        return self.cut

    def get_objective(self):
        return self.objective


class BdDnMessage(DnMessage):
    def __init__(self, solution: List[float], objective: float = 0.0) -> None:
        self.solution = solution
        self.objective = objective

    def get_solution(self):
        return self.solution

    def get_objective(self):
        return self.objective


class BdFinalDnMessage(FinalDnMessage):
    def __init__(self, solution: List[float] | None) -> None:
        self.solution = solution

    def get_solution(self) -> List[float] | None:
        return self.solution


class BdFinalUpMessage(FinalUpMessage):
    def __init__(self, objective: float | None) -> None:
        self.objective = objective

    def set_objective(self, obj: float) -> None:
        self.objective = obj

    def get_objective(self) -> float | None:
        return self.objective

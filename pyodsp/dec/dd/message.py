from typing import List, Dict
from ..node._message import (
    InitDnMessage,
    InitUpMessage,
    FinalDnMessage,
    FinalUpMessage,
    DnMessage,
    UpMessage,
)
from pyodsp.alg.bm.cuts import Cut


class DdInitDnMessage(InitDnMessage):
    def __init__(self, coupling_matrix: List[Dict[int, float]]) -> None:
        self.coupling_matrix = coupling_matrix

    def get_coupling_matrix(self):
        return self.coupling_matrix

    def set_depth(self, depth: int) -> None:
        self.depth = depth

    def get_depth(self) -> int:
        return self.depth


class DdInitUpMessage(InitUpMessage):
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


class DdUpMessage(UpMessage):
    def __init__(self, cut: Cut) -> None:
        self.cut = cut

    def get_cut(self):
        return self.cut

    def get_objective(self) -> float:
        return 0.0


class DdDnMessage(DnMessage):
    def __init__(self, solution: List[float]) -> None:
        self.solution = solution

    def get_solution(self):
        return self.solution

    def get_objective(self) -> float:
        return 0.0


class DdFinalDnMessage(FinalDnMessage):
    def __init__(self, solution: List[float] | None) -> None:
        self.solution = solution

    def get_solution(self) -> List[float] | None:
        return self.solution


class DdFinalUpMessage(FinalUpMessage):
    def __init__(
        self, objective: float | None, solution: list[float] | None = None
    ) -> None:
        self.objective = objective
        self.solution = solution

    def get_objective(self) -> float | None:
        return self.objective

    def set_objective(self, obj: float) -> None:
        self.objective = obj

    def get_solution(self) -> list[float] | None:
        return self.solution

from typing import List
from ..node._message import (
    NodeIdx,
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
        self.origin: NodeIdx | None = None

    def get_is_minimize(self) -> bool:
        return self.is_minimize

    def set_depth(self, depth: int) -> None:
        self.depth = depth

    def get_depth(self) -> int:
        return self.depth

    def set_origin(self, idx: NodeIdx) -> None:
        self.origin = idx

    def get_origin(self) -> NodeIdx:
        assert self.origin is not None
        return self.origin


class BdInitUpMessage(InitUpMessage):
    def __init__(self) -> None:
        self.bound = None

    def set_bound(self, bound: float | None) -> None:
        self.bound = bound

    def get_bound(self) -> float | None:
        return self.bound


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

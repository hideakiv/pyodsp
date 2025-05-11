from typing import List
from ..node._message import InitDnMessage, FinalDnMessage, DnMessage, UpMessage
from pyodsp.alg.cuts import Cut


class BdInitDnMessage(InitDnMessage):
    def __init__(self, is_minimize: bool) -> None:
        self.is_minimize = is_minimize

    def get_is_minimize(self) -> bool:
        return self.is_minimize

    def set_depth(self, depth: int) -> None:
        self.depth = depth

    def get_depth(self) -> int:
        return self.depth


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

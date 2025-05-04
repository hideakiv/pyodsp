from typing import List, Dict
from ..run._message import InitMessage, FinalMessage, DnMessage, UpMessage
from pyodsp.alg.cuts import Cut


class DdInitMessage(InitMessage):
    def __init__(self, coupling_matrix: List[Dict[int, float]]) -> None:
        self.coupling_matrix = coupling_matrix

    def get_coupling_matrix(self):
        return self.coupling_matrix


class DdUpMessage(UpMessage):
    def __init__(self, cut: Cut) -> None:
        self.cut = cut

    def get_cut(self):
        return self.cut


class DdDnMessage(DnMessage):
    def __init__(self, solution: List[float]) -> None:
        self.solution = solution

    def get_solution(self):
        return self.solution


class DdFinalMessage(FinalMessage):
    def __init__(self, solution: List[float]) -> None:
        self.solution = solution

    def get_solution(self):
        return self.solution

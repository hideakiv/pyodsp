from typing import List, Dict
from ..run._message import InitMessage, FinalMessage, DnMessage, UpMessage


class DdInitMessage(InitMessage):
    def __init__(self, coupling_matrix: List[Dict[int, float]]) -> None:
        self.coupling_matrix = coupling_matrix

    def get_coupling_matrix(self):
        return self.coupling_matrix


class DdUpMessage(UpMessage):
    pass


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

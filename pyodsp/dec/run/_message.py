from abc import ABC, abstractmethod
from typing import Any, List, Dict

NodeIdx = Any


class IMessage(ABC):
    pass


class InitMessage(IMessage):
    pass


class UpMessage(IMessage):
    pass


class DnMessage(IMessage):
    pass


class FinalMessage(IMessage):
    pass


class BdInitMessage(InitMessage):
    pass


class BdUpMessage(UpMessage):
    pass


class BdDnMessage(DnMessage):
    pass


class BdFinalMessage(FinalMessage):
    pass


class DdInitMessage(InitMessage):
    def __init__(self, coupling_matrix: List[Dict[int, float]]) -> None:
        self.coupling_matrix = coupling_matrix

    def get_coupling_matrix(self):
        return self.coupling_matrix


class DdUpMessage(UpMessage):
    pass


class DdDnMessage(DnMessage):
    pass


class DdFinalMessage(FinalMessage):
    def __init__(self, solution: List[float]) -> None:
        self.solution = solution

    def get_solution(self):
        return self.solution

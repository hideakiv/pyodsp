from abc import ABC, abstractmethod
from typing import Any, List, Dict

NodeIdx = Any

class IMessage(ABC):
    pass

class BdInitMessage(IMessage):
    pass

class BdFinalMessage(IMessage):
    pass

class DdInitMessage(IMessage):
    def __init__(self, coupling_matrix: List[Dict[int, float]]) -> None:
        self.coupling_matrix = coupling_matrix

    def get_coupling_matrix(self):
        return self.coupling_matrix

class DdFinalMessage(IMessage):
    def __init__(self, solution: List[float]) -> None:
        self.solution = solution

    def get_solution(self):
        return self.solution
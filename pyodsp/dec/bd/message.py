from typing import List
from ..run._message import InitMessage, FinalMessage, DnMessage, UpMessage
from pyodsp.alg.cuts import Cut


class BdInitMessage(InitMessage):
    pass


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


class BdFinalMessage(FinalMessage):
    pass

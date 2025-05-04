from typing import List
from ..run._message import InitMessage, FinalMessage, DnMessage, UpMessage


class BdInitMessage(InitMessage):
    pass


class BdUpMessage(UpMessage):
    pass


class BdDnMessage(DnMessage):
    def __init__(self, solution: List[float]) -> None:
        self.solution = solution

    def get_solution(self):
        return self.solution


class BdFinalMessage(FinalMessage):
    pass

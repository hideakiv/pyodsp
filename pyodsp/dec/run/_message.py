from abc import ABC, abstractmethod
from typing import List, Dict

class IMessage(ABC):
    pass

class BdInitMessage(IMessage):
    pass

class DdInitMessage(IMessage):
    def __init__(self, coupling_matrix: List[Dict[int, float]]) -> None:
        self.coupling_matrix = coupling_matrix

    def get_coupling_matrix(self):
        return self.coupling_matrix
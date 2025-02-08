from abc import ABC, abstractmethod
from typing import List

from pyomo.core.base.var import VarData

from pyodec.alg.bm.cuts import CutList


class BdAlgRoot(ABC):

    @abstractmethod
    def get_vars(self) -> List[VarData]:
        pass

    @abstractmethod
    def build(self, subobj_bounds: List[float]) -> None:
        pass

    @abstractmethod
    def reset_iteration(self) -> None:
        pass

    @abstractmethod
    def solve(self) -> None:
        pass

    @abstractmethod
    def get_solution(self) -> List[float]:
        pass

    @abstractmethod
    def add_cuts(self, cuts_list: List[CutList]) -> bool:
        pass

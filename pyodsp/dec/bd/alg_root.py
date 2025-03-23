from abc import ABC, abstractmethod
from typing import List, Tuple

from pyomo.core.base.var import VarData

from pyodsp.alg.cuts import CutList
from ..node._alg import IAlgRoot


class BdAlgRoot(IAlgRoot, ABC):

    @abstractmethod
    def get_vars(self) -> List[VarData]:
        pass

    @abstractmethod
    def build(self, subobj_bounds: List[float]) -> None:
        pass

    @abstractmethod
    def run_step(self, cuts_list: List[CutList] | None) -> Tuple[int, List[float]]:
        pass

    @abstractmethod
    def add_cuts(self, cuts_list: List[CutList]) -> None:
        pass

    @abstractmethod
    def reset_iteration(self) -> None:
        pass

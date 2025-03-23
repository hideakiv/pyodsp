from abc import ABC, abstractmethod
from typing import List

from pyomo.core.base.var import VarData

from pyodsp.alg.cuts import CutList
from ..node._alg import IAlgRoot


class BdAlgRoot(IAlgRoot, ABC):

    @abstractmethod
    def get_vars(self) -> List[VarData]:
        pass

    @abstractmethod
    def add_cuts(self, cuts_list: List[CutList]) -> None:
        pass

    @abstractmethod
    def reset_iteration(self) -> None:
        pass

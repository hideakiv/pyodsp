from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

from pyomo.core.base.var import VarData

from pyodec.alg.cuts import CutList


class BdAlgRoot(ABC):

    @abstractmethod
    def get_vars(self) -> List[VarData]:
        pass

    @abstractmethod
    def build(self, subobj_bounds: List[float]) -> None:
        pass

    @abstractmethod
    def run_step(self, cuts_list: List[CutList] | None) -> List[float] | None:
        pass

    @abstractmethod
    def reset_iteration(self) -> None:
        pass

    @abstractmethod
    def save(self, dir: Path) -> None:
        pass

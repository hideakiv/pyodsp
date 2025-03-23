from abc import ABC, abstractmethod
from typing import List

from pyomo.core.base.var import VarData

from ..node._alg import IAlgRoot


class BdAlgRoot(IAlgRoot, ABC):

    @abstractmethod
    def get_vars(self) -> List[VarData]:
        pass

    @abstractmethod
    def reset_iteration(self) -> None:
        pass

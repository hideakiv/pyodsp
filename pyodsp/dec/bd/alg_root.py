from abc import ABC, abstractmethod
from typing import List

from pyomo.core.base.var import VarData

from ..node._alg import IAlgRoot


class BdAlgRoot(IAlgRoot, ABC):

    pass

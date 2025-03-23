from abc import ABC, abstractmethod
from typing import List

from pyodsp.alg.cuts import Cut

from ..node._alg import IAlgLeaf


class BdAlgLeaf(IAlgLeaf, ABC):

    @abstractmethod
    def fix_variables(self, values: List[float]) -> None:
        """Fix the variables to a specified value

        Args:
            vars: The variables to be fixed.
            values: The values to be set.
        """

    @abstractmethod
    def get_subgradient(self) -> Cut:
        pass


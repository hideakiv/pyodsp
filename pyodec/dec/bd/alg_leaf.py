from abc import ABC, abstractmethod
from typing import List

from pyodec.alg.bm.cuts import Cut


class BdAlgLeaf(ABC):

    @abstractmethod
    def build(self) -> None:
        pass

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

from abc import ABC, abstractmethod
from typing import Any

from pyodsp.alg.cuts import Cut

NodeIdx = Any


class IMessage(ABC):
    pass


class InitMessage(IMessage, ABC):
    pass


class UpMessage(IMessage, ABC):
    @abstractmethod
    def get_cut(self) -> Cut:
        pass


class DnMessage(IMessage, ABC):
    pass


class FinalMessage(IMessage, ABC):
    pass

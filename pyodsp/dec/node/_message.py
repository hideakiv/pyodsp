from abc import ABC, abstractmethod
from typing import Any

from pyodsp.alg.bm.cuts import Cut

NodeIdx = Any


class IMessage(ABC):
    pass


class InitDnMessage(IMessage, ABC):
    @abstractmethod
    def get_is_minimize(self) -> bool:
        pass

    @abstractmethod
    def set_depth(self, depth: int) -> None:
        pass

    @abstractmethod
    def get_depth(self) -> int:
        pass


class InitUpMessage(IMessage, ABC):
    @abstractmethod
    def set_bound(self, bound: float | None) -> None:
        pass

    @abstractmethod
    def get_bound(self) -> float | None:
        pass


class UpMessage(IMessage, ABC):
    @abstractmethod
    def get_cut(self) -> Cut:
        pass


class DnMessage(IMessage, ABC):
    pass


class FinalDnMessage(IMessage, ABC):
    pass


class FinalUpMessage(IMessage, ABC):
    @abstractmethod
    def get_objective(self) -> float | None:
        pass

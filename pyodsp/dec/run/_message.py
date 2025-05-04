from abc import ABC, abstractmethod
from typing import Any, List, Dict

NodeIdx = Any


class IMessage(ABC):
    pass


class InitMessage(IMessage, ABC):
    pass


class UpMessage(IMessage, ABC):
    pass


class DnMessage(IMessage, ABC):
    pass


class FinalMessage(IMessage, ABC):
    pass

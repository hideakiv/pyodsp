from abc import ABC
from typing import Any

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

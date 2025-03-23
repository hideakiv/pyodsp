from abc import ABC, abstractmethod
from pathlib import Path

class IAlg(ABC):
    @abstractmethod
    def save(self, dir: Path) -> None:
        pass

    @abstractmethod
    def is_minimize(self) -> bool:
        pass

class IAlgRoot(IAlg, ABC):
    pass

class IAlgLeaf(IAlg, ABC):
    pass
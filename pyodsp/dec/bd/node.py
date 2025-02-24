from pathlib import Path

from pyodsp.dec.node import Node


class BdNode(Node):
    def __init__(
        self,
        idx: int,
        parent: int | None = None,
    ) -> None:
        super().__init__(idx, parent=parent)

    def build(self):
        pass

    def save(self, dir: Path):
        pass

    def is_minimize(self) -> bool:
        return True

from pathlib import Path

from pyodsp.dec.node import Node


class DdNode(Node):
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

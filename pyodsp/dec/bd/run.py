from typing import List
from pathlib import Path
import logging

from .logger import BdLogger
from .message import BdDnMessage
from ..node._node import INode
from ..graph.tree import Tree


class BdRun:
    def __init__(self, nodes: List[INode], filedir: Path, level: int = logging.INFO):
        self.logger = BdLogger(level)
        self.graph = Tree(nodes, self.logger, filedir)

    def run(self, init_solution: List[float] | None = None) -> None:
        if init_solution is None:
            dn_message = None
        else:
            dn_message = BdDnMessage(init_solution)

        self.graph.run(dn_message)

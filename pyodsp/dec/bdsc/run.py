from typing import List
from pathlib import Path
import logging

from .logger import BdScLogger
from .message import BdScDnMessage
from ..node._node import INode
from ..graph.hub_and_spoke import HubAndSpoke


class BdScRun:
    def __init__(
        self,
        nodes: List[INode],
        filedir: Path,
        level: int = logging.INFO,
        max_iteration: int = 1000,
    ):
        self.logger = BdScLogger(level)
        self.graph = HubAndSpoke(nodes, self.logger, filedir, max_iteration)

    def run(self, init_solution: List[float] | None = None) -> None:
        if init_solution is None:
            dn_message = None
        else:
            dn_message = BdScDnMessage(init_solution)

        self.graph.run(dn_message)

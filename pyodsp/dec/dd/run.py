from typing import List
from pathlib import Path

from .logger import DdLogger
from .message import DdDnMessage
from ..node._node import INode
from ..graph.hub_and_spoke import HubAndSpoke


class DdRun:
    def __init__(self, nodes: List[INode], filedir: Path):
        self.logger = DdLogger()
        self.graph = HubAndSpoke(nodes, self.logger, filedir)

    def run(self, init_solution: List[float] | None = None) -> None:
        if init_solution is None:
            dn_message = DdDnMessage(
                [0.0 for _ in range(self.graph.get_num_root_vars())]
            )
        else:
            dn_message = DdDnMessage(init_solution)

        self.graph.run(dn_message)

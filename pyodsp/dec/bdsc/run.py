from typing import List
from pathlib import Path
import logging

from .logger import BdScLogger
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
        if init_solution is not None:
            raise NotImplementedError(
                "Benders decomposition with scaled cuts does not support an "
                "initial solution: a dn message here also carries the master's "
                "rho, its subproblem bounds and its cut set, none of which "
                "exist before the master has taken a step."
            )

        self.graph.run(None)

from typing import List
from pathlib import Path
from mpi4py import MPI
import logging

from .logger import BdLogger
from .message import BdDnMessage
from ..node._node import INode
from ..graph.hub_and_spoke_mpi import HubAndSpokeMpi


class BdRunMpi:
    def __init__(self, nodes: List[INode], filedir: Path, level: int = logging.INFO):
        self.logger = BdLogger(level)
        self.graph = HubAndSpokeMpi(nodes, self.logger, filedir)

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def run(self, init_solution: List[float] | None = None) -> None:
        if self.rank == 0:
            if init_solution is None:
                dn_message = None
            else:
                dn_message = BdDnMessage(init_solution)

            self.graph.run(dn_message)
        else:
            self.graph.run()

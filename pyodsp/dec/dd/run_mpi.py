from typing import List
from pathlib import Path
from mpi4py import MPI
import logging

from .logger import DdLogger
from .message import DdDnMessage
from ..node._node import INode
from ..graph.hub_and_spoke_mpi import HubAndSpokeMpi


class DdRunMpi:
    def __init__(self, nodes: List[INode], filedir: Path, level: int = logging.INFO):
        self.logger = DdLogger(level)
        self.graph = HubAndSpokeMpi(nodes, self.logger, filedir)

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def run(self, init_solution: List[float] | None = None) -> None:
        if self.rank == 0:
            if init_solution is None:
                dn_message = DdDnMessage(
                    [0.0 for _ in range(self.graph.get_num_root_vars())]
                )
            else:
                dn_message = DdDnMessage(init_solution)

            self.graph.run(dn_message)
        else:
            self.graph.run()

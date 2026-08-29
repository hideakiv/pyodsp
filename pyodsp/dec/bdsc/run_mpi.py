from typing import List
from pathlib import Path
from mpi4py import MPI
import logging

from ..node._node import INode
from ..node._logger import AlgLogger
from ..graph.hub_and_spoke_mpi import HubAndSpokeMpi


class BdScRunMpi:
    """Benders decomposition with scaled cuts, hub and spoke across ranks.

    Rank 0 holds the root and drives the master; every other rank holds a
    share of the leaves and runs their column generation. Build only the
    nodes belonging to this rank — the same split BdRunMpi/DdRunMpi use.

    Each trial point the root broadcasts carries its whole cut set, which
    every subproblem mirrors (see BdScAlgLeafPyomo), so the message is
    larger here than in plain Benders; it is sent only on the steps where
    the master's cuts actually changed.
    """

    def __init__(
        self,
        nodes: List[INode],
        filedir: Path,
        level: int = logging.INFO,
        max_iteration: int = 1000,
    ):
        self.logger = AlgLogger("Benders decomposition with scaled cuts", "bdsc", level)
        self.graph = HubAndSpokeMpi(nodes, self.logger, filedir, max_iteration)

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def run(self, init_solution: List[float] | None = None) -> None:
        if init_solution is not None:
            raise NotImplementedError(
                "Benders decomposition with scaled cuts does not support an "
                "initial solution: a dn message here also carries the master's "
                "rho, its subproblem bounds and its cut set, none of which "
                "exist before the master has taken a step."
            )

        self.graph.run(None)

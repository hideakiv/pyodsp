from typing import List
from pathlib import Path
import logging

from .logger import SddpLogger
from ..bd.message import BdDnMessage
from ..node._node import INode
from ..graph.lattice_mpi import LatticeMpi


class SddpRunMpi:
    """Parallelized SDDP — see LatticeMpi for the rank protocol.

    Build the exact same `nodes` on every rank (unlike BdRunMpi/DdRunMpi,
    which split distinct nodes across ranks): only the alg_root objects
    should differ, with rank 0 using the default (purging) CutsManager and
    every other rank constructed with purgeable=False.
    """

    def __init__(
        self,
        nodes: List[List[INode]],
        filedir: Path,
        level: int = logging.INFO,
        max_iteration: int = 1000,
        sample_frequency: int = 10,
        sample_size: int = 1000,
        confidence_level: float = 0.95,
    ):
        self.logger = SddpLogger(level)
        self.graph = LatticeMpi(
            nodes,
            self.logger,
            filedir,
            max_iteration,
            sample_frequency,
            sample_size,
            confidence_level,
        )

    def run(self, init_solution: List[float] | None = None) -> None:
        if init_solution is None:
            dn_message = None
        else:
            dn_message = BdDnMessage(init_solution)

        self.graph.run(dn_message)

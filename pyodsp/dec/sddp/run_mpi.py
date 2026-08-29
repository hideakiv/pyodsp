from typing import List
from pathlib import Path
import logging

from ..node._node import INode
from ..node._logger import AlgLogger
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
        self.logger = AlgLogger("SDDP", "sddp", level)
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
        if init_solution is not None:
            raise NotImplementedError(
                "SDDP does not support an initial solution: every iteration's "
                "forward pass begins by solving the root, which sets the "
                "first-stage state itself, so there is nowhere to inject one."
            )

        self.graph.run()

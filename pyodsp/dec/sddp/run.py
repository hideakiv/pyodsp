from typing import List
from pathlib import Path
import logging

from .logger import SddpLogger
from ..node._node import INode
from ..graph.lattice import Lattice


class SddpRun:
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
        self.graph = Lattice(
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

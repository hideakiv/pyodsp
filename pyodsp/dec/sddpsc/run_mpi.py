from typing import List
from pathlib import Path
import logging

from ..node._node import INode
from ..node._logger import AlgLogger
from ..graph.lattice_mpi import LatticeMpi

__all__ = ["SddpScLatticeMpi", "SddpScRunMpi"]


class SddpScLatticeMpi(LatticeMpi):
    """LatticeMpi for sddpsc's masters, whose state is more than cuts.

    The rank protocol is plain SDDP's: rank 0 runs every forward and
    backward pass, and the other ranks only evaluate their slice of the
    Monte Carlo sample. What a replica is sent before each round is the
    whole of SddpScAlgRootBm.get_snapshot -- cuts and rho -- so its masters
    hand their leaves the same dn messages rank 0's would.

    Nothing else needs to travel. An inner node's leaf role mirrors its
    root role's cuts onto its future term on every pass_dn_message (see
    SddpScNodeInner), and a sample path's cost at a leaf is the recourse
    at the trial point, solved with the state fixed -- it does not depend
    on the columns the leaf's column generation has collected, which do
    differ between ranks.
    """

    def _snapshot(self, node: INode):
        return node.alg_root.get_snapshot()

    def _apply_snapshot(self, node: INode, snapshot) -> None:
        node.alg_root.apply_snapshot(snapshot)


class SddpScRunMpi:
    """Parallelized SDDP with scaled cuts -- see SddpScLatticeMpi.

    As with SddpRunMpi, build the exact same `nodes` on every rank, with
    rank 0's SddpScAlgRootBm masters purgeable (the default) and every
    other rank's built with purgeable=False.
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
        gap_tolerance: float | None = None,
        stable_tolerance: float | None = None,
        stall_iterations: int | None = None,
        stall_tolerance: float | None = None,
    ):
        self.logger = AlgLogger("SDDP with scaled cuts", "sddpsc", level)
        self.graph = SddpScLatticeMpi(
            nodes,
            self.logger,
            filedir,
            max_iteration,
            sample_frequency,
            sample_size,
            confidence_level,
            gap_tolerance,
            stable_tolerance,
            stall_iterations,
            stall_tolerance,
        )

    def run(self, init_solution: List[float] | None = None) -> None:
        if init_solution is not None:
            raise NotImplementedError(
                "SDDP does not support an initial solution: every iteration's "
                "forward pass begins by solving the root, which sets the "
                "first-stage state itself, so there is nowhere to inject one."
            )

        self.graph.run()

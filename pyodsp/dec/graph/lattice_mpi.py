from pathlib import Path
from typing import Dict, List

import numpy as np
from mpi4py import MPI

from .lattice import Lattice
from ..node._logger import ILogger
from ..node._node import INode
from ..node._message import NodeIdx


class LatticeMpi(Lattice):
    """Parallelized SDDP.

    Unlike the other ``*_mpi.py`` graphs, every rank holds an identical copy
    of the whole lattice rather than a distinct slice of it — only rank 0
    ever generates cuts (runs the trunk forward pass and the backward
    pass). Every other rank exists solely to help evaluate the Monte Carlo
    upper-bound simulation (Lattice._collect_samples) in parallel, since
    that loop only depends on the current, frozen set of cuts and is
    otherwise embarrassingly parallel.

    Build the exact same `nodes` structure on every rank. The one
    required difference: rank 0's alg_root objects must use the default
    (purging) CutsManager, while every other rank's must be constructed
    with purgeable=False (see BdAlgRootBm) — a non-root rank must never
    decide independently that a cut is stale or dominated, since it isn't
    solving the same trial points rank 0 is.

    Before every Monte Carlo round, rank 0 broadcasts its current, exact
    set of cuts per node (BundleMethod.get_cut_list) and every replica
    wholesale-replaces its own cuts with it (BundleMethod.replace_cuts,
    force=True). Replaying rank 0's individual add_cuts calls instead
    would not work: whether a cut survives depends on a dominance check
    against theta's *current solved value*, which rank 0 updates via
    interleaved solves a replica never performs — so an independent replay
    would silently diverge from rank 0's actual surviving set even though
    every cut it started from was one rank 0 also saw.

    Rank 0 also relays the root's current solution (its dn_message from
    the last _run_root, which fixes every stage-1 node's coupling
    variable) — since only rank 0 ever calls _run_root, a replica's
    stage-1 nodes would otherwise still be solving against whatever
    (unfixed) coupling value they started with. Every deeper stage is
    unaffected: _run_forwards's own traversal re-establishes each sampled
    node's children itself as it descends.
    """

    def __init__(
        self,
        nodes: List[List[INode]],
        logger: ILogger,
        filedir: Path,
        max_iteration: int = 1000,
        sample_frequency: int = 10,
        sample_size: int = 1000,
        confidence_level: float = 0.95,
    ) -> None:
        super().__init__(
            nodes,
            logger,
            filedir,
            max_iteration,
            sample_frequency,
            sample_size,
            confidence_level,
        )
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def run(self) -> None:
        if self.rank == 0:
            self.logger.log_initialization()
        self._run_init()
        self._run_main()
        self._save()

    def _run_main(self) -> None:
        if self.rank == 0:
            self._run_main_root()
        else:
            self._run_worker_loop()

    def _run_main_root(self) -> None:
        if self.root is None:
            raise ValueError("Root node not found")
        bound = -1e9
        for iteration in range(self.max_iteration):
            bound = self._run_root()
            if iteration % self.sample_frequency == self.sample_frequency - 1:
                self._broadcast_sync()
                if self._termination(bound):
                    break
            else:
                self._run_forwards(self._iteration_rng(iteration))

            bound = self._run_backwards()

        self._broadcast_stop()

    def _run_worker_loop(self) -> None:
        while True:
            message = self.comm.bcast(None, root=0)
            if message == "stop":
                break
            self._apply_snapshots(message["snapshots"])
            self._apply_root_dn_message(message["root_dn_message"])
            self._collect_samples()

    def _broadcast_sync(self) -> None:
        if self.root is None:
            raise ValueError("Root node not found")
        snapshots: Dict[NodeIdx, list] = {}
        for stage in range(self.num_stages - 1):
            for node_idx in self.stages[stage]:
                node = self.nodes[node_idx]
                snapshots[node_idx] = node.alg_root.bm.get_cut_list()
        root_dn_message = self._last_dn_messages[self.root.get_idx()]
        self.comm.bcast(
            {"snapshots": snapshots, "root_dn_message": root_dn_message}, root=0
        )

    def _broadcast_stop(self) -> None:
        self.comm.bcast("stop", root=0)

    def _apply_snapshots(self, snapshots: Dict[NodeIdx, list]) -> None:
        for node_idx, cuts_list in snapshots.items():
            node = self.nodes[node_idx]
            node.alg_root.bm.replace_cuts(cuts_list)

    def _apply_root_dn_message(self, dn_message) -> None:
        if self.root is None:
            raise ValueError("Root node not found")
        for child_id in self.root.get_children():
            child = self.nodes[child_id]
            child.pass_dn_message(dn_message)

    def _collect_samples(self) -> List[float]:
        size = self.comm.Get_size()
        indices = np.array_split(np.arange(self.sample_size), size)[self.rank]
        my_objectives = []
        for i in indices:
            # keyed by the global sample index, so this rank's slice draws
            # exactly the streams it would have drawn sequentially
            my_objectives.append(self._run_forwards(self._sample_rng(int(i))))
        gathered = self.comm.gather(my_objectives, root=0)
        if self.rank == 0:
            return [obj for sub in gathered for obj in sub]
        return []

    def _save(self) -> None:
        if self.rank == 0:
            super()._save()

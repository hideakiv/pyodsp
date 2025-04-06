from typing import List, Dict
from pathlib import Path
from mpi4py import MPI

from pyodsp.alg.const import *

from .run import BdRun
from ..node._node import INode, INodeLeaf, INodeInner


class BdRunMpi(BdRun):
    def __init__(self, nodes: List[INode], filedir: Path):
        super().__init__(nodes, filedir)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        for node in nodes:
            if isinstance(node, INodeInner):
                raise ValueError("Nested Benders not supported in BdRunMpi")

        # gather node-rank info
        id_list = [node.get_idx() for node in nodes]
        all_ids = self.comm.gather({self.rank: id_list}, root=0)
        self.node_rank_map: Dict[int, int] = {}
        if self.rank == 0:
            for all_id in all_ids:
                for target, ids in all_id.items():
                    for idx in ids:
                        self.node_rank_map[idx] = target

    def run(self):
        bounds = {}
        for node in self.nodes.values():
            if isinstance(node, INodeLeaf):
                bounds[node.get_idx()] = node.get_bound()

        all_bounds = self.comm.gather(bounds, root=0)

        if self.rank == 0:
            assert self.root is not None
            is_minimize = self.root.is_minimize()
            self.root.set_depth(0)
            self.root.set_logger()
            self.comm.bcast(is_minimize, root=0)
            depth = self.root.get_depth()
            self.comm.bcast(depth, root=0)
            for node in self.nodes.values():
                self._init_leaf(node, is_minimize, depth + 1)
            self._run_root(all_bounds)
        else:
            is_minimize = None
            is_minimize = self.comm.bcast(is_minimize, root=0)
            depth = None
            depth = self.comm.bcast(depth, root=0)
            for node in self.nodes.values():
                self._init_leaf(node, is_minimize, depth + 1)
            self._run_leaf()

        for node in self.nodes.values():
            node.save(self.filedir)

    def _init_leaf(self, node: INode, is_minimize: bool, depth: int) -> None:
        if isinstance(node, INodeLeaf):
            if node.is_minimize() != is_minimize:
                raise ValueError("Inconsistent optimization sense")
            node.set_depth(depth)

    def _run_root(self, all_bounds) -> None:
        assert self.root is not None
        self.logger.log_initialization()
        combined_bounds = {}
        for d in all_bounds:
            combined_bounds.update(d)
        for child in self.root.get_children():
            self.root.set_child_bound(child, combined_bounds[child])
        self.root.build()

        self.root.reset()
        combined_cuts_dn = None
        while True:
            status, solution = self.root.run_step(combined_cuts_dn)

            if status != STATUS_NOT_FINISHED:
                self.comm.bcast(-1, root=0)
                return

            self.comm.bcast(solution, root=0)

            cuts_dn = {}
            for node in self.nodes.values():
                if isinstance(node, INodeLeaf):
                    cut_dn = self._get_cut(node.get_idx(), solution)
                    cuts_dn[node.get_idx()] = cut_dn

            all_cuts_dn = self.comm.gather(cuts_dn, root=0)
            combined_cuts_dn = {}
            for d in all_cuts_dn:
                combined_cuts_dn.update(d)

    def _run_leaf(self) -> None:
        solution = None
        while True:
            solution = self.comm.bcast(solution, root=0)
            if solution == -1:
                return None

            cuts_dn = {}
            for node in self.nodes.values():
                if isinstance(node, INodeLeaf):
                    cut_dn = self._get_cut(node.get_idx(), solution)
                    cuts_dn[node.get_idx()] = cut_dn

            all_cuts_dn = self.comm.gather(cuts_dn, root=0)

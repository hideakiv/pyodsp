from typing import List, Dict
from pathlib import Path
from mpi4py import MPI

from pyodsp.alg.const import *

from .run import BdRun
from ..node._node import INode, INodeLeaf, INodeInner
from ..run._message import InitMessage


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
            self.root.set_depth(0)
            self.root.set_logger()
            init_message = self.root.get_init_message()
            self.comm.bcast(init_message, root=0)
            for node in self.nodes.values():
                self._init_leaf(node, init_message)
            self._run_root(all_bounds)
        else:
            init_message = None
            init_message = self.comm.bcast(init_message, root=0)
            for node in self.nodes.values():
                self._init_leaf(node, init_message)
            self._run_leaf()

        for node in self.nodes.values():
            node.save(self.filedir)

    def _init_leaf(self, node: INode, init_message: InitMessage) -> None:
        if isinstance(node, INodeLeaf):
            node.pass_init_message(init_message)

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
        combined_up_messages = None
        while True:
            status, new_dn_message = self.root.run_step(combined_up_messages)

            if status != STATUS_NOT_FINISHED:
                self.comm.bcast(-1, root=0)
                return

            self.comm.bcast(new_dn_message, root=0)

            up_messages = {}
            for node in self.nodes.values():
                if isinstance(node, INodeLeaf):
                    up_message = self._get_up_message(node.get_idx(), new_dn_message)
                    up_messages[node.get_idx()] = up_message

            all_up_messages = self.comm.gather(up_messages, root=0)
            combined_up_messages = {}
            for d in all_up_messages:
                combined_up_messages.update(d)

    def _run_leaf(self) -> None:
        dn_message = None
        while True:
            dn_message = self.comm.bcast(dn_message, root=0)
            if dn_message == -1:
                return None

            up_messages = {}
            for node in self.nodes.values():
                if isinstance(node, INodeLeaf):
                    up_message = self._get_up_message(node.get_idx(), dn_message)
                    up_messages[node.get_idx()] = up_message

            all_up_messages = self.comm.gather(up_messages, root=0)

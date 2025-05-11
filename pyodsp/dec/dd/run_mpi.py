from typing import List, Dict
from pathlib import Path
from mpi4py import MPI

from pyodsp.alg.const import *

from .run import DdRun
from .message import DdDnMessage
from ..node._node import INode
from ..node._message import InitMessage, FinalMessage, DnMessage, UpMessage


class DdRunMpi(DdRun):
    def __init__(self, nodes: List[INode], filedir: Path):
        super().__init__(nodes, filedir)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        # gather node-rank info
        id_list = [node.get_idx() for node in nodes]
        all_ids = self.comm.gather({self.rank: id_list}, root=0)
        self.node_rank_map: Dict[int, int] = {}
        if self.rank == 0:
            for all_id in all_ids:
                for target, ids in all_id.items():
                    for idx in ids:
                        self.node_rank_map[idx] = target

    def run(self, init_solution: List[float] | None = None):
        if self.rank == 0:
            assert self.root is not None
            self.root.set_depth(0)
            self.root.set_logger()
            self._init_root()

            init_messages = self._split_init_messages()

            for target, init_message in init_messages.items():
                self.comm.send(init_message, dest=target, tag=0)

            if 0 in init_messages:
                sub_init_messages = init_messages[0]
                for node_id, init_message in sub_init_messages.items():
                    self._init_leaf(node_id, init_message)

            self._run_root(init_solution)

            self._finalize_root()
        else:
            init_messages = self.comm.recv(source=0, tag=0)

            for node_id, init_message in init_messages.items():
                self._init_leaf(node_id, init_message)

            self._run_leaf_mpi()

            self._finalize_leaf_mpi()

        for node in self.nodes.values():
            node.save(self.filedir)

    def _split_init_messages(self) -> Dict[int, Dict[int, InitMessage]]:
        init_messages: Dict[int, Dict[int, InitMessage]] = {}
        assert self.root is not None
        for child_id in self.root.get_children():
            target = self.node_rank_map[child_id]
            if target not in init_messages:
                init_messages[target] = {}
            init_messages[target][child_id] = self.root.get_init_message(
                child_id=child_id
            )
        return init_messages

    def _run_root(self, init_solution: List[float] | None = None) -> None:
        assert self.root is not None
        self.root.reset()
        if init_solution is None:
            dn_message = DdDnMessage([0.0 for _ in range(self.root.get_num_vars())])
        else:
            dn_message = DdDnMessage(init_solution)

        # broadcast solution
        self.comm.bcast(dn_message, root=0)

        up_messages = self._run_leaf(dn_message)

        # gather cuts
        all_up_messages = self.comm.gather(up_messages, root=0)
        combined_up_messages = {}
        for d in all_up_messages:
            combined_up_messages.update(d)

        while True:
            status, new_dn_message = self.root.run_step(combined_up_messages)
            if status != STATUS_NOT_FINISHED:
                self.comm.bcast(-1, root=0)
                break
            # broadcast solution
            self.comm.bcast(new_dn_message, root=0)

            up_messages = self._run_leaf(new_dn_message)

            # gather cuts
            all_up_messages = self.comm.gather(up_messages, root=0)
            combined_up_messages = {}
            for d in all_up_messages:
                combined_up_messages.update(d)

        self.logger.log_finaliziation()

    def _run_leaf_mpi(self) -> None:
        message: DnMessage = None
        message = self.comm.bcast(message, root=0)
        up_messages = self._run_leaf(message)
        all_up_messages = self.comm.gather(up_messages, root=0)
        while True:
            message = self.comm.bcast(message, root=0)
            if message == -1:
                break
            up_messages = self._run_leaf(message)
            all_up_messages = self.comm.gather(up_messages, root=0)

    def _finalize_root(self) -> None:
        assert self.root is not None

        # split solutions
        solutions_dict: Dict[int, Dict[int, FinalMessage]] = {}
        for child_id in self.root.get_children():
            target = self.node_rank_map[child_id]
            if target not in solutions_dict:
                solutions_dict[target] = {}
            message = self.root.get_final_message(
                node_id=target, groups=self.root.get_groups()
            )
            solutions_dict[target][child_id] = message

        for target, messages in solutions_dict.items():
            self.comm.send(messages, dest=target, tag=1)

        final_obj = 0.0
        if 0 in solutions_dict:
            for node_id, message in solutions_dict[0].items():
                sub_obj = self._finalize_leaf(node_id, message)
                final_obj += sub_obj
        all_objs = self.comm.gather(final_obj, root=0)
        total_obj = 0.0
        for objval in all_objs:
            total_obj += objval
        self.logger.log_completion(total_obj)

    def _finalize_leaf_mpi(self):
        solutions_info = self.comm.recv(source=0, tag=1)
        final_obj = 0.0
        for node_id, message in solutions_info.items():
            sub_obj = self._finalize_leaf(node_id, message)
            final_obj += sub_obj
        all_objs = self.comm.gather(final_obj, root=0)

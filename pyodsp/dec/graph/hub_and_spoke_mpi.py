from pathlib import Path
from typing import List, Dict
from mpi4py import MPI

from .hub_and_spoke import HubAndSpoke
from ..node._logger import ILogger
from ..node._node import INode
from ..node._message import (
    InitDnMessage,
    InitUpMessage,
    DnMessage,
    UpMessage,
    FinalDnMessage,
    FinalUpMessage,
    NodeIdx,
)

from pyodsp.alg.const import STATUS_NOT_FINISHED


class HubAndSpokeMpi(HubAndSpoke):
    def __init__(self, nodes: List[INode], logger: ILogger, filedir: Path) -> None:
        super().__init__(nodes, logger, filedir)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        self._gather_node_rank_map()

    def _gather_node_rank_map(self) -> None:
        # gather node-rank info
        id_list = [node.get_idx() for node in self.leaves]
        all_ids: List[Dict[int, List[NodeIdx]]] = self.comm.gather(
            {self.rank: id_list}, root=0
        )
        self.node_rank_map: Dict[NodeIdx, int] = {}
        if self.rank == 0:
            assert self.root is not None
            self.node_rank_map[self.root.get_idx()] = self.rank
            for all_id in all_ids:
                for rank, ids in all_id.items():
                    for idx in ids:
                        self.node_rank_map[idx] = rank

    def run(self, init_solution: DnMessage | None = None):
        if self.rank == 0:
            self.logger.log_initialization()
            self._run_init_dn()
            self._run_init_up()
            up_messages = self._run_main_preprocess(init_solution)
            self._run_main(up_messages)
            self.logger.log_finaliziation()
            final_obj = self._run_final()
            self.logger.log_completion(final_obj)
            self._save()
        else:
            self._run_init_dn_mpi()
            self._run_init_up_mpi()
            self._run_main_preprocess_mpi()
            self._run_main_mpi()
            self._run_final_mpi()
            self._save_mpi()

    def _init_root(self) -> None:
        super()._init_root()
        assert self.root is not None

        init_messages: Dict[int, Dict[NodeIdx, InitDnMessage]] = {}
        for child_id in self.root.get_children():
            target = self.node_rank_map[child_id]
            if target == 0:
                continue
            if target not in init_messages:
                init_messages[target] = {}
            init_messages[target][child_id] = self.root.get_init_dn_message(
                child_id=child_id
            )

        for target, init_message in init_messages.items():
            self.comm.send(init_message, dest=target, tag=0)

    def _run_init_dn_mpi(self) -> None:
        init_messages = self.comm.recv(source=0, tag=0)

        for leaf in self.leaves:
            init_message = init_messages[leaf.get_idx()]
            self._init_leaf(leaf, init_message)

    def _run_init_up_core(self) -> Dict[NodeIdx, InitUpMessage]:
        # gather messages
        up_messages = super()._run_init_up_core()
        all_up_messages = self.comm.gather(up_messages, root=0)
        combined_up_messages = {}
        for d in all_up_messages:
            if d is None:
                continue
            combined_up_messages.update(d)
        return combined_up_messages

    def _run_init_up_mpi(self) -> None:
        up_messages = {}
        for leaf in self.leaves:
            init_message = leaf.get_init_up_message()
            up_messages[leaf.get_idx()] = init_message
        all_up_messages = self.comm.gather(up_messages, root=0)

    def _run_main_preprocess(
        self, init_solution: DnMessage | None
    ) -> Dict[NodeIdx, UpMessage] | None:
        # broadcast solution
        self.comm.bcast(init_solution, root=0)
        up_messages = super()._run_main_preprocess(init_solution)

        # gather cuts
        all_up_messages = self.comm.gather(up_messages, root=0)
        if init_solution is None:
            return None
        combined_up_messages = {}
        for d in all_up_messages:
            if d is None:
                continue
            combined_up_messages.update(d)
        return combined_up_messages

    def _run_main_preprocess_mpi(self) -> None:
        message: DnMessage = None
        message = self.comm.bcast(message, root=0)
        if message is None:
            up_messages = None
        else:
            up_messages = self._run_leaf(message)
        all_up_messages = self.comm.gather(up_messages, root=0)

    def _run_main(self, up_messages: Dict[NodeIdx, UpMessage] | None) -> None:
        combined_up_messages = up_messages
        while True:
            status, dn_message = self._run_root(combined_up_messages)
            if status != STATUS_NOT_FINISHED:
                self.comm.bcast(-1, root=0)
                break
            # broadcast solution
            self.comm.bcast(dn_message, root=0)

            up_messages = self._run_leaf(dn_message)

            # gather cuts
            all_up_messages = self.comm.gather(up_messages, root=0)
            combined_up_messages = {}
            for d in all_up_messages:
                combined_up_messages.update(d)

    def _run_main_mpi(self) -> None:
        message: DnMessage = None
        while True:
            message = self.comm.bcast(message, root=0)
            if message == -1:
                break
            up_messages = self._run_leaf(message)
            all_up_messages = self.comm.gather(up_messages, root=0)

    def _run_final_core(self) -> Dict[NodeIdx, FinalUpMessage]:
        up_messages = super()._run_final_core()
        all_up_messages = self.comm.gather(up_messages, root=0)
        combined_up_messages = {}
        for d in all_up_messages:
            if d is None:
                continue
            combined_up_messages.update(d)
        return combined_up_messages

    def _finalize_root(self) -> None:
        if self.root is None:
            raise ValueError("root node not found")

        # split solutions
        solutions_dict: Dict[int, Dict[int, FinalDnMessage]] = {}
        for child_id in self.root.get_children():
            target = self.node_rank_map[child_id]
            if target not in solutions_dict:
                solutions_dict[target] = {}
            message = self.root.get_final_dn_message(
                node_id=target, groups=self.root.get_groups()
            )
            solutions_dict[target][child_id] = message

        for target, messages in solutions_dict.items():
            self.comm.send(messages, dest=target, tag=1)

    def _run_final_mpi(self) -> None:
        messages = self.comm.recv(source=0, tag=1)
        up_messages = {}
        for leaf in self.leaves:
            dn_message = messages[leaf.get_idx()]
            up_message = self._finalize_leaf(leaf, dn_message)
            up_messages[leaf.get_idx()] = up_message
        all_objs = self.comm.gather(up_messages, root=0)

    def _save_mpi(self) -> None:
        for node in self.leaves:
            self._save_leaf(node)

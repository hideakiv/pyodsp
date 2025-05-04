from typing import List, Dict
from pathlib import Path
from mpi4py import MPI

from pyodsp.alg.const import *

from .run import DdRun
from .message import DdDnMessage
from .mip_heuristic_root import MipHeuristicRoot
from ..node._node import INode
from ..run._message import InitMessage, FinalMessage, DnMessage, UpMessage


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
            is_minimize = self.root.is_minimize()
            self.comm.bcast(is_minimize, root=0)
            self.comm.bcast(self.root.get_depth(), root=0)

            matrices = self._split_matrices()

            for target, matrix in matrices.items():
                self.comm.send(matrix, dest=target, tag=0)

            if 0 in matrices:
                matrix_info = matrices[0]
                for node_id, matrix in matrix_info.items():
                    self._init_leaf(
                        node_id, matrix, is_minimize, self.root.get_depth() + 1
                    )

            self._run_root(init_solution)

            self._finalize_root()
        else:
            is_minimize = None
            is_minimize = self.comm.bcast(is_minimize, root=0)
            depth = None
            depth = self.comm.bcast(depth, root=0)
            matrix_info = self.comm.recv(source=0, tag=0)

            for node_id, matrix in matrix_info.items():
                self._init_leaf(node_id, matrix, is_minimize, depth + 1)

            self._run_leaf_mpi()

            self._finalize_leaf_mpi()

        for node in self.nodes.values():
            node.save(self.filedir)

    def _split_matrices(self) -> Dict[int, Dict[int, InitMessage]]:
        matrices: Dict[int, Dict[int, InitMessage]] = {}
        assert self.root is not None
        for child_id in self.root.get_children():
            target = self.node_rank_map[child_id]
            if target not in matrices:
                matrices[target] = {}
            matrices[target][child_id] = self.root.get_init_message(child_id=child_id)
        return matrices

    def _run_root(self, init_solution: List[float] | None = None) -> None:
        assert self.root is not None
        self.root.reset()
        if init_solution is None:
            dn_message = DdDnMessage([0.0 for _ in range(self.root.get_num_vars())])
        else:
            dn_message = DdDnMessage(init_solution)

        # broadcast solution
        self.comm.bcast(dn_message, root=0)

        cuts_dn = self._run_leaf(dn_message)

        # gather cuts
        all_cuts_dn = self.comm.gather(cuts_dn, root=0)
        combined_cuts_dn = {}
        for d in all_cuts_dn:
            combined_cuts_dn.update(d)

        while True:
            status, new_dn_message = self.root.run_step(combined_cuts_dn)
            if status != STATUS_NOT_FINISHED:
                self.comm.bcast(-1, root=0)
                break
            # broadcast solution
            self.comm.bcast(new_dn_message, root=0)

            cuts_dn = self._run_leaf(new_dn_message)

            # gather cuts
            all_cuts_dn = self.comm.gather(cuts_dn, root=0)
            combined_cuts_dn = {}
            for d in all_cuts_dn:
                combined_cuts_dn.update(d)

        self.logger.log_finaliziation()

    def _run_leaf_mpi(self) -> None:
        message: DnMessage = None
        message = self.comm.bcast(message, root=0)
        cuts_dn = self._run_leaf(message)
        all_cuts_dn = self.comm.gather(cuts_dn, root=0)
        while True:
            message = self.comm.bcast(message, root=0)
            if message == -1:
                break
            cuts_dn = self._run_leaf(message)
            all_cuts_dn = self.comm.gather(cuts_dn, root=0)

    def _finalize_root(self) -> None:
        assert self.root is not None
        mip_heuristic = MipHeuristicRoot(
            self.root.get_groups(), self.root.get_alg_root(), **self.root.get_kwargs()
        )
        mip_heuristic.build()
        solutions = mip_heuristic.run()

        # split solutions
        solutions_dict: Dict[int, Dict[int, FinalMessage]] = {}
        for child_id in self.root.get_children():
            target = self.node_rank_map[child_id]
            if target not in solutions_dict:
                solutions_dict[target] = {}
            solutions_dict[target][child_id] = solutions[child_id]

        for target, message in solutions_dict.items():
            self.comm.send(message, dest=target, tag=1)

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

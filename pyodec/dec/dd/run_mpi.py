from typing import List, Dict
from mpi4py import MPI

from .run import DdRun
from .node import DdNode
from .node_leaf import DdLeafNode
from .node_root import DdRootNode


class DdRunMpi(DdRun):
    def __init__(
        self,
        nodes: List[DdNode],
        node_rank_map: Dict[int, int],
    ):
        super().__init__(nodes)
        self.node_rank_map = node_rank_map
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def run(self):
        matrix_info: Dict[int, List[Dict[int, float]]] = {}
        if self.rank == 0:
            self.logger.log_initialization()
            root = self.nodes[self.root_idx]
            assert isinstance(root, DdRootNode)
            root.build()
            matrices: Dict[int, Dict[int, List[Dict[int, float]]]] = {}
            for child_id in root.get_children():
                target = self.node_rank_map[child_id]
                if not target in matrices:
                    matrices[target] = {}
                matrices[target][child_id] = root.lagrangian_data.matrix[child_id]

            for target, matrix in matrices.items():
                self.comm.send(matrix, dest=target, tag=0)

            if 0 in matrices:
                matrix_info = matrices[0]
                for node in self.nodes.values():
                    if isinstance(node, DdLeafNode):
                        node_id = node.idx
                        node.set_coupling_matrix(matrix_info[node_id])

            root.alg.reset_iteration()
            solution = [0.0 for _ in range(root.num_constrs)]
            self.comm.bcast(solution, root=0)
            cuts_dn = self._run_leaf(solution)
            all_cuts_dn = self.comm.gather(cuts_dn, root=0)
            combined_cuts_dn = {}
            for d in all_cuts_dn:
                combined_cuts_dn.update(d)
            finished = root.add_cuts(combined_cuts_dn)
            while True:
                if finished:
                    self.comm.bcast(-1, root=0)
                    return None
                solution = self._run_root(root)
                self.comm.bcast(solution, root=0)
                cuts_dn = self._run_leaf(solution)
                all_cuts_dn = self.comm.gather(cuts_dn, root=0)
                combined_cuts_dn = {}
                for d in all_cuts_dn:
                    combined_cuts_dn.update(d)
                finished = root.add_cuts(combined_cuts_dn)
        else:
            matrix_info = self.comm.recv(source=0, tag=0)

            for node in self.nodes.values():
                if isinstance(node, DdLeafNode):
                    node_id = node.idx
                    node.set_coupling_matrix(matrix_info[node_id])

            solution: List[float] = None
            solution = self.comm.bcast(solution, root=0)
            cuts_dn = self._run_leaf(solution)
            all_cuts_dn = self.comm.gather(cuts_dn, root=0)
            while True:
                solution = self.comm.bcast(solution, root=0)
                if solution == -1:
                    return None
                cuts_dn = self._run_leaf(solution)
                all_cuts_dn = self.comm.gather(cuts_dn, root=0)

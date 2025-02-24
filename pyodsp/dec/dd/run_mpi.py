from typing import List, Dict
from pathlib import Path
from mpi4py import MPI

from .run import DdRun
from .node import DdNode
from .node_leaf import DdLeafNode
from .node_root import DdRootNode


class DdRunMpi(DdRun):
    def __init__(
        self, nodes: List[DdNode], node_rank_map: Dict[int, int], filedir: Path
    ):
        super().__init__(nodes, filedir)
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
                if target not in matrices:
                    matrices[target] = {}
                matrices[target][child_id] = root.alg.lagrangian_data.matrix[child_id]

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
            while True:
                solution = root.run_step(combined_cuts_dn)
                if solution is None:
                    self.comm.bcast(-1, root=0)
                    break
                self.comm.bcast(solution, root=0)

                cuts_dn = self._run_leaf(solution)
                all_cuts_dn = self.comm.gather(cuts_dn, root=0)
                combined_cuts_dn = {}
                for d in all_cuts_dn:
                    combined_cuts_dn.update(d)
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
                    break
                cuts_dn = self._run_leaf(solution)
                all_cuts_dn = self.comm.gather(cuts_dn, root=0)

        if self.rank == 0:
            self.logger.log_finaliziation()
            root = self.nodes[self.root_idx]
            assert isinstance(root, DdRootNode)

            solutions = root.solve_mip_heuristic()
            solutions_dict: Dict[int, Dict[int, List[float]]] = {}
            for child_id in root.get_children():
                target = self.node_rank_map[child_id]
                if target not in solutions_dict:
                    solutions_dict[target] = {}
                solutions_dict[target][child_id] = solutions[child_id]

            for target, sols in solutions_dict.items():
                self.comm.send(sols, dest=target, tag=1)
            
            final_obj = 0.0
            if 0 in solutions_dict:
                for node in self.nodes.values():
                    if isinstance(node, DdLeafNode):
                        node.alg.fix_variables_and_solve(solutions[node.idx])
                    final_obj += node.alg.get_objective_value()
            all_objs = self.comm.gather(final_obj, root=0)
            total_obj = 0.0
            for objval in all_objs:
                total_obj += objval
            self.logger.log_completion(total_obj)
        else:
            solutions_info = self.comm.recv(source=0, tag=1)
            final_obj = 0.0
            for node in self.nodes.values():
                if isinstance(node, DdLeafNode):
                    node.alg.fix_variables_and_solve(solutions_info[node.idx])
                    final_obj += node.alg.get_objective_value()
            all_objs = self.comm.gather(final_obj, root=0)
            

        for node in self.nodes.values():
            node.save(self.filedir)

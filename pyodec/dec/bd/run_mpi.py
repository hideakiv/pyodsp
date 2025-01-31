from typing import List, Dict
from mpi4py import MPI

from .run import BdRun
from .node import BdNode
from .node_leaf import BdLeafNode
from .node_root import BdRootNode


class BdRunMpi(BdRun):
    def __init__(
        self,
        nodes: List[BdNode],
        node_rank_map: Dict[int, int],
    ):
        super().__init__(nodes)
        self.node_rank_map = node_rank_map
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def run(self):
        bounds = {}
        for node in self.nodes.values():
            if isinstance(node, BdLeafNode):
                bounds[node.idx] = node.get_bound()

        all_bounds = self.comm.gather(bounds, root=0)

        if self.rank == 0:
            self.logger.log_initialization()
            combined_bounds = {}
            for d in all_bounds:
                combined_bounds.update(d)
            root = self.nodes[self.root_idx]
            assert isinstance(root, BdRootNode)
            for child in root.children:
                root.set_bound(child, combined_bounds[child])
            root.build()

            root.solver.reset_iteration()
            while True:
                root.solve()
                solution = root.get_coupling_solution()

                obj = self.get_root_obj()
                assert obj is not None
                self.relax_bound.append(obj)
                self.logger.log_master_problem(len(self.relax_bound), obj, solution)
                self.comm.bcast(solution, root=0)

                cuts_dn = {}
                for node in self.nodes.values():
                    if isinstance(node, BdLeafNode):
                        cut_dn = self._get_cut(node.idx, solution)
                        cuts_dn[node.idx] = cut_dn

                all_cuts_dn = self.comm.gather(cuts_dn, root=0)
                combined_cuts_dn = {}
                for d in all_cuts_dn:
                    combined_cuts_dn.update(d)
                finished = root.add_cuts(combined_cuts_dn)
                if finished:
                    self.logger.log_completion(
                        len(self.relax_bound), self.relax_bound[-1]
                    )
                    if isinstance(root, BdLeafNode):
                        raise NotImplementedError()
                    else:
                        self.comm.bcast(-1, root=0)
                        return None

        else:
            solution = None
            while True:
                solution = self.comm.bcast(solution, root=0)
                if solution == -1:
                    return None

                cuts_dn = {}
                for node in self.nodes.values():
                    if isinstance(node, BdLeafNode):
                        cut_dn = self._get_cut(node.idx, solution)
                        cuts_dn[node.idx] = cut_dn

                all_cuts_dn = self.comm.gather(cuts_dn, root=0)

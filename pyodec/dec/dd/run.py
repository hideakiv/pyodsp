from typing import List, Dict

from numpy import isin

from pyodec.alg.bm.cuts import Cut, OptimalityCut, FeasibilityCut

from .logger import DdLogger
from .node import DdNode
from .node_leaf import DdLeafNode
from .node_root import DdRootNode


class DdRun:
    def __init__(self, nodes: List[DdNode]):
        self.nodes: Dict[int, DdNode] = {node.idx: node for node in nodes}
        self.root_idx = self._get_root_idx()
        self.logger = DdLogger()
        self.relax_bound: List[float] = []

    def _get_root_idx(self) -> int | None:
        for idx, node in self.nodes.items():
            if node.parent is None:
                return idx
        return None

    def get_root_obj(self) -> float | None:
        if self.root_idx is not None:
            node = self.nodes[self.root_idx]
            assert isinstance(node, DdRootNode)
            return node.solver.get_objective_value()
        return None

    def run(self):
        self.logger.log_initialization()
        root = self.nodes[self.root_idx]
        assert isinstance(root, DdRootNode)
        root.build()
        for child_id in root.get_children():
            child = self.nodes[child_id]
            assert isinstance(child, DdLeafNode)
            child.set_coupling_matrix(root.lagrangian_data.matrix[child_id])

        root.solver.reset_iteration()
        cuts_dn = self._run_leaf([0.0 for _ in range(root.num_constrs)])
        finished = root.add_cuts(cuts_dn)
        while True:
            if finished:
                self.logger.log_completion(len(self.relax_bound), self.relax_bound[-1])
                return None
            solution = self._run_root(root)
            cuts_dn = self._run_leaf(solution)
            finished = root.add_cuts(cuts_dn)

    def _run_root(self, root: DdRootNode) -> List[float]:
        root.solve()
        solution = root.get_dual_solution()
        obj = self.get_root_obj()
        assert obj is not None
        self.relax_bound.append(obj)
        self.logger.log_master_problem(len(self.relax_bound), obj, solution)

        return solution

    def _run_leaf(self, solution: List[float]) -> Dict[int, Cut]:
        cuts_dn = {}
        for node in self.nodes.values():
            if isinstance(node, DdLeafNode):
                cut_dn = self._get_cut(node.idx, solution)
                cuts_dn[node.idx] = cut_dn
        return cuts_dn

    def _get_cut(self, idx: int, solution: List[float]) -> Cut:
        node = self.nodes[idx]
        assert isinstance(node, DdLeafNode)
        node.build()
        cut_dn = node.solve(solution)
        assert cut_dn is not None
        if isinstance(cut_dn, OptimalityCut):
            self.logger.log_sub_problem(idx, "Optimality", cut_dn.coeffs, cut_dn.rhs)
        if isinstance(cut_dn, FeasibilityCut):
            self.logger.log_sub_problem(idx, "Feasibility", cut_dn.coeffs, cut_dn.rhs)
        return cut_dn

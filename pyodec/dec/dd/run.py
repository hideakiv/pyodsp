from typing import List, Dict

from pyodec.alg.cuts import Cut, OptimalityCut, FeasibilityCut

from .logger import DdLogger
from .node import DdNode
from .node_leaf import DdLeafNode
from .node_root import DdRootNode


class DdRun:
    def __init__(self, nodes: List[DdNode]):
        self.nodes: Dict[int, DdNode] = {node.idx: node for node in nodes}
        self.root_idx = self._get_root_idx()
        self.logger = DdLogger()

    def _get_root_idx(self) -> int | None:
        for idx, node in self.nodes.items():
            if node.parent is None:
                return idx
        return None

    def run(self):
        self.logger.log_initialization()
        root = self.nodes[self.root_idx]
        assert isinstance(root, DdRootNode)
        root.build()
        for child_id in root.get_children():
            child = self.nodes[child_id]
            assert isinstance(child, DdLeafNode)
            child.set_coupling_matrix(root.alg.lagrangian_data.matrix[child_id])

        root.alg.reset_iteration()
        cuts_dn = self._run_leaf([0.0 for _ in range(root.num_constrs)])
        while True:
            solution = root.run_step(cuts_dn)
            if solution is None:
                return
            cuts_dn = self._run_leaf(solution)

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

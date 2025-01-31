from typing import List, Dict

from pyodec.alg.bm.cuts import Cut, OptimalityCut, FeasibilityCut

from .logger import BdLogger
from .node import BdNode
from .node_leaf import BdLeafNode
from .node_root import BdRootNode


class BdRun:
    def __init__(self, nodes: List[BdNode]):
        self.nodes: Dict[int, BdNode] = {node.idx: node for node in nodes}
        self.root_idx = self._get_root_idx()
        self.logger = BdLogger()
        self.relax_bound: List[float] = []

    def _get_root_idx(self) -> int | None:
        for idx, node in self.nodes.items():
            if node.parent is None:
                return idx
        return None

    def get_root_obj(self) -> float | None:
        if self.root_idx is not None:
            node = self.nodes[self.root_idx]
            assert isinstance(node, BdRootNode)
            return node.solver.get_objective_value()
        return None

    def run(self):
        if self.root_idx is not None:
            self.logger.log_initialization()
            self._run_node(self.nodes[self.root_idx])

    def _run_node(self, node: BdNode, sol_up: List[float] | None = None) -> Cut | None:
        if isinstance(node, BdRootNode):
            self._set_bounds(node)
            node.build()

            node.solver.reset_iteration()
            while True:
                if isinstance(node, BdLeafNode):
                    cut_up = node.solve(sol_up)
                else:
                    cut_up = node.solve()
                solution = node.get_coupling_solution()

                if node.idx == self.root_idx:
                    obj = self.get_root_obj()
                    assert obj is not None
                    self.relax_bound.append(obj)
                    self.logger.log_master_problem(len(self.relax_bound), obj, solution)
                cuts_dn = self._get_cuts(node, solution)
                finished = node.add_cuts(cuts_dn)
                if finished:
                    self.logger.log_completion(
                        len(self.relax_bound), self.relax_bound[-1]
                    )
                    if isinstance(node, BdLeafNode):
                        return cut_up
                    else:
                        return None
        if isinstance(node, BdLeafNode):
            node.build()
            return node.solve(sol_up)

    def _get_cuts(self, node: BdNode, solution: List[float]) -> Dict[int, Cut]:
        cuts_dn = {}
        for child in node.children:
            cut_dn = self._get_cut(child, solution)
            cuts_dn[child] = cut_dn
        return cuts_dn

    def _get_cut(self, idx: int, solution: List[float]) -> Cut:
        cut_dn = self._run_node(self.nodes[idx], solution)
        assert cut_dn is not None
        if isinstance(cut_dn, OptimalityCut):
            self.logger.log_sub_problem(idx, "Optimality", cut_dn.coeffs, cut_dn.rhs)
        if isinstance(cut_dn, FeasibilityCut):
            self.logger.log_sub_problem(idx, "Feasibility", cut_dn.coeffs, cut_dn.rhs)
        return cut_dn

    def _set_bounds(self, node: BdRootNode) -> None:
        for child in node.children:
            child_node = self.nodes[child]
            assert isinstance(child_node, BdLeafNode)
            node.set_bound(child, child_node.get_bound())

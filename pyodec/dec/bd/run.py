from typing import List, Dict

from .logger import BdLogger
from .node import BdNode
from .node_leaf import BdLeafNode
from .node_root import BdRootNode
from .cuts import Cut, OptimalityCut, FeasibilityCut


class BdRun:
    def __init__(self, nodes: List[BdNode], tolerance=1e-6, max_iteration=1000):
        self.nodes: Dict[int, BdNode] = {node.idx: node for node in nodes}
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.root_idx = self._get_root_idx()
        self.logger = BdLogger()
        self.iteration = 0
        self.lb: List[float] = []

    def _get_root_idx(self) -> int:
        for idx, node in self.nodes.items():
            if node.parent is None:
                return idx
        return None

    def get_root_obj(self) -> float | None:
        if self.root_idx is not None:
            return self.nodes[self.root_idx].solver.get_objective_value()
        return None

    def run(self):
        if self.root_idx is not None:
            self.logger.log_initialization(self.tolerance, self.max_iteration)
            self._run_node(self.nodes[self.root_idx])

    def _run_node(self, node: BdNode, sol_up: List[float] | None = None) -> Cut | None:
        if isinstance(node, BdRootNode):
            self._set_bounds(node)
            if not node.built:
                node.build()
            node.set_tolerance(self.tolerance)

            while self.iteration < self.max_iteration:
                if isinstance(node, BdLeafNode):
                    cut_up = node.solve(sol_up)
                else:
                    node.solve()
                solution = node.get_coupling_solution()

                if node.idx == self.root_idx:
                    self.iteration += 1
                    obj = self.get_root_obj()
                    self.lb.append(obj)
                    self.logger.log_master_problem(self.iteration, obj, solution)
                cuts_dn = self._get_cuts(node, solution)
                optimal = node.add_cuts(cuts_dn)
                if optimal:
                    self.logger.log_completion(self.iteration, self.lb[-1])
                    if isinstance(node, BdLeafNode):
                        return cut_up
                    else:
                        return None
        if isinstance(node, BdLeafNode):
            if not node.built:
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
        if isinstance(cut_dn, OptimalityCut):
            self.logger.log_sub_problem(
                idx, "Optimality", cut_dn.coefficients, cut_dn.constant
            )
        if isinstance(cut_dn, FeasibilityCut):
            self.logger.log_sub_problem(
                idx, "Feasibility", cut_dn.coefficients, cut_dn.constant
            )
        return cut_dn

    def _set_bounds(self, node: BdRootNode) -> None:
        for child in node.children:
            node.set_bound(child, self.nodes[child].get_bound())

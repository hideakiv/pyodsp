from typing import List, Dict

from .node import BdNode
from .node_leaf import BdLeafNode
from .node_root import BdRootNode
from .cuts import Cut


class BdRun:
    def __init__(self, nodes: List[BdNode]):
        self.nodes: Dict[int, BdNode] = {node.idx: node for node in nodes}
        self.root_idx = self._get_root()

    def _get_root(self) -> int:
        for idx, node in self.nodes.items():
            if node.parent is None:
                return idx
        return None

    def get_root_obj(self) -> float:
        return self.nodes[self.root_idx].solver.get_objective_value()

    def run(self):
        if self.root_idx is not None:
            if not self.nodes[self.root_idx].built:
                self.nodes[self.root_idx].build()
            self._iterate(self.nodes[self.root_idx])

    def _iterate(self, node: BdNode, sol_up: List[float] | None = None) -> Cut | None:
        if isinstance(node, BdRootNode):
            while True:
                if isinstance(node, BdLeafNode):
                    cut_up = node.solve(sol_up)
                else:
                    node.solve()
                solution = node.get_coupling_solution()

                print(self.get_root_obj(), solution)
                cuts_dn = {}
                for child in node.children:
                    cut_dn = self._iterate(self.nodes[child], solution)
                    cuts_dn[child] = cut_dn
                optimal = node.add_cuts(cuts_dn)
                if optimal:
                    if isinstance(node, BdLeafNode):
                        return cut_up
                    else:
                        return None
        if isinstance(node, BdLeafNode):
            return node.solve(sol_up)

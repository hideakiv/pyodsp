from typing import List, Dict
from pathlib import Path

from pyodsp.alg.cuts import Cut, OptimalityCut, FeasibilityCut
from pyodsp.alg.const import *

from .logger import BdLogger
from ..utils import create_directory
from ..node._node import INode, INodeRoot, INodeLeaf, INodeInner


class BdRun:
    def __init__(self, nodes: List[INode], filedir: Path):
        self.nodes: Dict[int, INode] = {node.get_idx(): node for node in nodes}
        self.root = self._get_root()
        self.logger = BdLogger()

        self.filedir = filedir
        create_directory(self.filedir)

    def _get_root(self) -> INodeRoot | None:
        for node in self.nodes.values():
            if isinstance(node, INodeRoot) and not isinstance(node, INodeLeaf):
                return node
        return None

    def run(self):
        if self.root is not None:
            self.logger.log_initialization()
            self.root.set_depth(0)
            self._run_check(self.root)
            self._run_node(self.root)
            self._run_finalize(self.root)
        for node in self.nodes.values():
            node.save(self.filedir)

    def _run_check(self, node: INode) -> None:
        if isinstance(node, INodeRoot):
            node.set_logger()
        for child_id in node.get_children():
            child = self.nodes[child_id]
            child.set_depth(node.get_depth() + 1)
            if child.is_minimize() != node.is_minimize():
                raise ValueError("Inconsistent optimization sense")
            self._run_check(child)

    def _run_node(self, node: INode, sol_up: List[float] | None = None) -> Cut | None:
        if isinstance(node, INodeRoot):
            self._set_bounds(node)
            node.build()
            if isinstance(node, INodeInner):
                assert sol_up is not None
                node.pass_solution(sol_up)

            node.reset()
            cuts_dn = None
            while True:
                status, solution = node.run_step(cuts_dn)

                if status != STATUS_NOT_FINISHED:
                    if isinstance(node, INodeInner):
                        if (
                            status == STATUS_MAX_ITERATION
                            or status == STATUS_TIME_LIMIT
                        ):
                            cuts_dn = self._get_cuts(node, solution)
                            node.add_cuts(cuts_dn)
                        return node.get_subgradient()
                    else:
                        return

                cuts_dn = self._get_cuts(node, solution)
        if isinstance(node, INodeLeaf):
            assert sol_up is not None
            node.build()
            return node.solve(sol_up)

    def _run_finalize(self, node: INode) -> None:
        if isinstance(node, INodeRoot):
            solution = node.get_solution_dn()
            for child_id in node.get_children():
                child = self.nodes[child_id]
                if isinstance(child, INodeLeaf):
                    child.solve(solution)
                elif isinstance(child, INodeInner):
                    child.pass_solution(solution)
                    child.get_subgradient()
                self._run_finalize(child)

    def _get_cuts(self, node: INode, solution: List[float]) -> Dict[int, Cut]:
        cuts_dn = {}
        for child in node.get_children():
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

    def _set_bounds(self, node: INodeRoot) -> None:
        for child in node.get_children():
            child_node = self.nodes[child]
            assert isinstance(child_node, INodeLeaf)
            node.set_child_bound(child, child_node.get_bound())

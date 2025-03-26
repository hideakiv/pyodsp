from typing import List, Dict
from pathlib import Path

from pyodsp.alg.cuts import Cut, OptimalityCut, FeasibilityCut
from pyodsp.alg.const import *

from .logger import DdLogger
from .mip_heuristic_root import MipHeuristicRoot
from ..utils import create_directory
from ..node._node import INode, INodeRoot, INodeLeaf
from ..run._message import IMessage


class DdRun:
    def __init__(self, nodes: List[INode], filedir: Path):
        self.nodes: Dict[int, INode] = {node.get_idx(): node for node in nodes}
        self.root = self._get_root()
        self.logger = DdLogger()

        self.filedir = filedir
        create_directory(self.filedir)

    def _get_root(self) -> INodeRoot | None:
        for node in self.nodes.values():
            if isinstance(node, INodeRoot) and not isinstance(node, INodeLeaf):
                return node
        return None

    def run(self):
        if self.root is not None:
            # run root process
            self.root.set_depth(0)
            self.root.set_logger()
            self._init_root()
            for child_id in self.root.get_children():
                self._init_leaf(
                    child_id, 
                    self.root.get_init_message(child_id=child_id), 
                    self.root.is_minimize(),
                    self.root.get_depth() + 1
                )

            self._run_root()
            self._finalize_root()
        else:
            raise ValueError("root node not found")

        for node in self.nodes.values():
            node.save(self.filedir)
    
    def _init_root(self) -> None:
        self.logger.log_initialization()
        assert self.root is not None
        self.root.build()

    def _init_leaf(
            self, node_id: int, 
            message: IMessage, 
            is_minimize: bool,
            depth: int,
        ) -> None:
        node = self.nodes[node_id]
        node.set_depth(depth)
        assert isinstance(node, INodeLeaf)
        if node.is_minimize() != is_minimize:
            raise ValueError("Inconsistent optimization sense")
        node.pass_init_message(message)

    def _run_root(self) -> None:
        assert self.root is not None
        self.root.reset()
        cuts_dn = self._run_leaf([0.0 for _ in range(self.root.get_num_vars())])
        while True:
            status, solution = self.root.run_step(cuts_dn)
            if status != STATUS_NOT_FINISHED:
                break
            cuts_dn = self._run_leaf(solution)

        self.logger.log_finaliziation()

    def _run_leaf(self, solution: List[float]) -> Dict[int, Cut]:
        cuts_dn = {}
        for node in self.nodes.values():
            if isinstance(node, INodeLeaf):
                cut_dn = self._get_cut(node.get_idx(), solution)
                cuts_dn[node.get_idx()] = cut_dn
        return cuts_dn

    def _get_cut(self, idx: int, solution: List[float]) -> Cut:
        node = self.nodes[idx]
        assert isinstance(node, INodeLeaf)
        node.build()
        cut_dn = node.solve(solution)
        assert cut_dn is not None
        if isinstance(cut_dn, OptimalityCut):
            self.logger.log_sub_problem(idx, "Optimality", cut_dn.coeffs, cut_dn.rhs)
        if isinstance(cut_dn, FeasibilityCut):
            self.logger.log_sub_problem(idx, "Feasibility", cut_dn.coeffs, cut_dn.rhs)
        return cut_dn
    
    def _finalize_root(self) -> None:
        assert self.root is not None
        mip_heuristic = MipHeuristicRoot(
            self.root.get_groups(), self.root.get_alg_root(), **self.root.get_kwargs()
        )
        mip_heuristic.build()
        solutions = mip_heuristic.run()
        final_obj = 0.0
        for node_id, message in solutions.items():
            sub_obj = self._finalize_leaf(node_id, message)
            final_obj += sub_obj
        self.logger.log_completion(final_obj)

    def _finalize_leaf(self, node_id, message: IMessage) -> float:
        node = self.nodes[node_id]
        assert isinstance(node, INodeLeaf)
        node.pass_final_message(message)
        return node.get_objective_value()


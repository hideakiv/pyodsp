from typing import List, Dict
from pathlib import Path

from pyodsp.alg.cuts import Cut, OptimalityCut, FeasibilityCut

from .logger import DdLogger
from .node import DdNode
from .node_leaf import DdLeafNode
from .node_root import DdRootNode
from ..utils import create_directory, SparseMatrix


class DdRun:
    def __init__(self, nodes: List[DdNode], filedir: Path):
        self.nodes: Dict[int, DdNode] = {node.idx: node for node in nodes}
        self.root = self._get_root()
        self.logger = DdLogger()

        self.filedir = filedir
        create_directory(self.filedir)

    def _get_root(self) -> DdRootNode | None:
        for node in self.nodes.values():
            if node.parent is None:
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
                    self.root.alg.lagrangian_data.matrix[child_id], 
                    self.root.is_minimize,
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
        self.root.build()

    def _init_leaf(
            self, node_id: int, 
            matrix: SparseMatrix, 
            is_minimize: bool,
            depth: int,
        ) -> None:
        node = self.nodes[node_id]
        node.set_depth(depth)
        assert isinstance(node, DdLeafNode)
        if node.is_minimize != is_minimize:
            raise ValueError("Inconsistent optimization sense")
        node.set_coupling_matrix(matrix)

    def _run_root(self) -> None:
        self.root.alg.reset_iteration()
        cuts_dn = self._run_leaf([0.0 for _ in range(self.root.num_constrs)])
        while True:
            solution = self.root.run_step(cuts_dn)
            if solution is None:
                break
            cuts_dn = self._run_leaf(solution)

        self.logger.log_finaliziation()

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
    
    def _finalize_root(self) -> None:
        solutions = self.root.solve_mip_heuristic()
        final_obj = 0.0
        for node_id, sols in solutions.items():
            sub_obj = self._finalize_leaf(node_id, sols)
            final_obj += sub_obj
        self.logger.log_completion(final_obj)

    def _finalize_leaf(self, node_id, solution: List[float]) -> float:
        node = self.nodes[node_id]
        assert isinstance(node, DdLeafNode)
        node.alg.fix_variables_and_solve(solution)
        return node.alg.get_objective_value()


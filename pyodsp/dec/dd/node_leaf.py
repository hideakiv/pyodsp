from typing import List, Dict

from pyodsp.alg.cuts import Cut, OptimalityCut, FeasibilityCut
from pyodsp.alg.params import DEC_CUT_ABS_TOL

from ..node.dec_node import DecNodeLeaf
from .alg_leaf import DdAlgLeaf
from .coupling_manager import CouplingManager


class DdLeafNode(DecNodeLeaf):
    def __init__(
        self,
        idx: int,
        alg_leaf: DdAlgLeaf,
        parent: int,
    ) -> None:
        super().__init__(idx, alg_leaf)
        self.add_parent(parent)
        self._is_minimize = alg_leaf.is_minimize()
        self.len_vars = alg_leaf.get_len_vars()

    def set_coupling_matrix(self, coupling_matrix: List[Dict[int, float]]) -> None:
        self.cm = CouplingManager(coupling_matrix, self.len_vars, self.is_minimize())

    def solve(self, dual_values: List[float]) -> Cut:
        primal_coeffs = self.cm.dual_times_matrix(dual_values)
        self.alg_leaf.update_objective(primal_coeffs)
        is_optimal, solution, obj = self.alg_leaf.get_solution_or_ray()
        if is_optimal:
            dual_coeffs = self.cm.matrix_times_primal(solution)
            product = self.cm.inner_product(primal_coeffs, solution)
            rhs = obj - product
            sparse_coeff = {j: val for j, val in enumerate(dual_coeffs) if abs(val) > DEC_CUT_ABS_TOL}
            return OptimalityCut(
                coeffs=sparse_coeff,
                rhs=rhs,
                objective_value=obj,
                info={"solution": solution},
            )
        else:
            dual_coeffs = self.cm.matrix_times_primal(solution)
            product = self.cm.inner_product(primal_coeffs, solution)
            rhs = obj - product
            sparse_coeff = {j: val for j, val in enumerate(dual_coeffs) if abs(val) > DEC_CUT_ABS_TOL}
            return FeasibilityCut(
                coeffs=sparse_coeff, rhs=rhs, info={"solution": solution}
            )
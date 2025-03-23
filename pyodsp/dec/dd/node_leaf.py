from typing import List, Dict

from pyodsp.alg.cuts import Cut, OptimalityCut, FeasibilityCut
from pyodsp.alg.params import DEC_CUT_ABS_TOL

from ..node.dec_node import DecNodeLeaf
from .alg_leaf import DdAlgLeaf


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
        self.row_major: List[Dict[int, float]] = coupling_matrix
        self.len_constrs = len(coupling_matrix)
        self.col_major: List[Dict[int, float]] = self._convert_to_col_major(
            coupling_matrix
        )

    def solve(self, dual_values: List[float]) -> Cut:
        primal_coeffs = self._dual_times_matrix(dual_values)
        self.alg_leaf.update_objective(primal_coeffs)
        is_optimal, solution, obj = self.alg_leaf.get_solution_or_ray()
        if is_optimal:
            dual_coeffs = self._matrix_times_primal(solution)
            product = self._inner_product(primal_coeffs, solution)
            rhs = obj - product
            sparse_coeff = {j: val for j, val in enumerate(dual_coeffs) if abs(val) > DEC_CUT_ABS_TOL}
            return OptimalityCut(
                coeffs=sparse_coeff,
                rhs=rhs,
                objective_value=obj,
                info={"solution": solution},
            )
        else:
            dual_coeffs = self._matrix_times_primal(solution)
            product = self._inner_product(primal_coeffs, solution)
            rhs = obj - product
            sparse_coeff = {j: val for j, val in enumerate(dual_coeffs) if abs(val) > DEC_CUT_ABS_TOL}
            return FeasibilityCut(
                coeffs=sparse_coeff, rhs=rhs, info={"solution": solution}
            )

    def _convert_to_col_major(
        self, row_major: List[Dict[int, float]]
    ) -> List[Dict[int, float]]:
        cols: List[Dict[int, float]] = [{} for _ in range(self.len_vars)]
        for i, row in enumerate(row_major):
            for j, val in row.items():
                cols[j][i] = val
        return cols

    def _dual_times_matrix(self, dual_values: List[float]) -> List[float]:
        # multiply dual_values and coupling_matrix
        coeffs = [0.0] * self.len_vars
        for j, col in enumerate(self.col_major):
            coeff = 0.0
            for i, val in col.items():
                coeff += dual_values[i] * val
            if self.is_minimize():
                coeffs[j] = coeff
            else:
                coeffs[j] = -coeff
        return coeffs

    def _matrix_times_primal(self, primal_values: List[float]) -> List[float]:
        # multiply coupling_matrix and primal
        coeffs = [0.0] * self.len_constrs
        for i, row in enumerate(self.row_major):
            coeff = 0.0
            for j, val in row.items():
                coeff += val * primal_values[j]
            if self.is_minimize():
                coeffs[i] = -coeff
            else:
                coeffs[i] = coeff
        return coeffs

    def _inner_product(self, x: List[float], y: List[float]) -> float:
        product = 0.0
        for xval, yval in zip(x, y):
            product += xval * yval

        return product

from typing import List, Dict
from pathlib import Path

from pyodsp.alg.cuts import Cut, OptimalityCut, FeasibilityCut

from .node import DdNode
from .alg_leaf import DdAlgLeaf
from ..utils import create_directory


class DdLeafNode(DdNode):
    def __init__(
        self,
        idx: int,
        alg: DdAlgLeaf,
        parent: int,
    ) -> None:
        super().__init__(idx, parent=parent)
        self.alg = alg
        self.is_minimize = self.alg.is_minimize()
        self.len_vars = self.alg.get_len_vars()

        self.built = False

    def set_coupling_matrix(self, coupling_matrix: List[Dict[int, float]]) -> None:
        self.row_major: List[Dict[int, float]] = coupling_matrix
        self.len_constrs = len(coupling_matrix)
        self.col_major: List[Dict[int, float]] = self._convert_to_col_major(
            coupling_matrix
        )

    def build(self) -> None:
        if self.built:
            return
        self.alg.build()
        self.built = True

    def solve(self, dual_values: List[float]) -> Cut:
        primal_coeffs = self._dual_times_matrix(dual_values)
        self.alg.update_objective(primal_coeffs)
        is_optimal, solution, obj = self.alg.get_solution_or_ray()
        if is_optimal:
            dual_coeffs = self._matrix_times_primal(solution)
            product = self._inner_product(primal_coeffs, solution)
            rhs = obj - product
            return OptimalityCut(
                coeffs=dual_coeffs,
                rhs=rhs,
                objective_value=obj,
                info={"solution": solution},
            )
        else:
            dual_coeffs = self._matrix_times_primal(solution)
            product = self._inner_product(primal_coeffs, solution)
            rhs = obj - product
            return FeasibilityCut(
                coeffs=dual_coeffs, rhs=rhs, info={"solution": solution}
            )

    def save(self, dir: Path):
        node_dir = dir / f"node{self.idx}"
        create_directory(node_dir)
        self.alg.save(node_dir)

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
            if self.is_minimize:
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
            if self.is_minimize:
                coeffs[i] = -coeff
            else:
                coeffs[i] = coeff
        return coeffs

    def _inner_product(self, x: List[float], y: List[float]) -> float:
        product = 0.0
        for xval, yval in zip(x, y):
            product += xval * yval

        return product

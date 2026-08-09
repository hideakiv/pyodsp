from typing import List, Dict


class CouplingManager:
    def __init__(self, coupling_matrix: List[Dict[int, float]], len_vars: int) -> None:
        self.len_vars = len_vars
        self.row_major: List[Dict[int, float]] = coupling_matrix
        self.len_constrs = len(coupling_matrix)
        self.col_major: List[Dict[int, float]] = self._convert_to_col_major(
            coupling_matrix
        )

    def _convert_to_col_major(
        self, row_major: List[Dict[int, float]]
    ) -> List[Dict[int, float]]:
        cols: List[Dict[int, float]] = [{} for _ in range(self.len_vars)]
        for i, row in enumerate(row_major):
            for j, val in row.items():
                cols[j][i] = val
        return cols

    def dual_times_matrix(self, dual_values: List[float]) -> List[float]:
        # multiply dual_values and coupling_matrix
        coeffs = [0.0] * self.len_vars
        for j, col in enumerate(self.col_major):
            coeff = 0.0
            for i, val in col.items():
                coeff += dual_values[i] * val
            coeffs[j] = coeff
        return coeffs

    def matrix_times_primal(self, primal_values: List[float]) -> List[float]:
        # multiply coupling_matrix and primal
        coeffs = [0.0] * self.len_constrs
        for i, row in enumerate(self.row_major):
            coeff = 0.0
            for j, val in row.items():
                coeff += val * primal_values[j]
            coeffs[i] = -coeff
        return coeffs

    def inner_product(self, x: List[float], y: List[float]) -> float:
        product = 0.0
        for xval, yval in zip(x, y):
            product += xval * yval

        return product

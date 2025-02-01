from typing import List, Dict, Tuple

from pyomo.core.base.var import VarData

from pyodec.alg.bm.cuts import Cut, OptimalityCut, FeasibilityCut

from .node import DdNode
from .solver_leaf import DdSolverLeaf


class DdLeafNode(DdNode):
    def __init__(
        self,
        idx: int,
        solver: DdSolverLeaf,
        parent: int,
        vars_up: List[VarData],
    ) -> None:
        super().__init__(idx, parent=parent)
        self.solver = solver
        self.coupling_vars_up: List[VarData] = vars_up
        self.is_minimize = self.solver.get_objective_sense()

        self.built = False

    def set_coupling_matrix(self, coupling_matrix: List[Dict[int, float]]) -> None:
        self.row_major: List[Dict[int, float]]
        self.col_major: List[Dict[int, float]]

    def build(self) -> None:
        if self.built:
            return
        self.solver.build()
        self.built = True

    def solve(self, dual_values: List[float]) -> Cut:
        primal_coeffs = self._dual_times_matrix(dual_values)
        self.solver.add_to_objective(self.coupling_vars_up, primal_coeffs)
        self.solver.solve()
        is_optimal, solution = self.solver.get_solution_or_ray(self.coupling_vars_up)
        if is_optimal:
            dual_coeffs = self._matrix_times_primal(solution)
            product = self._inner_product(primal_coeffs, solution)
            obj = self.solver.get_objective_value()
            rhs = obj - product
            return OptimalityCut(dual_coeffs, rhs, obj)
        else:
            NotImplementedError()

    def _dual_times_matrix(self, dual_values: List[float]) -> List[float]:
        # multiply dual_values and coupling_matrix
        coeffs = [0.0] * len(self.coupling_vars_up)
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
        coeffs = [0.0] * len(self.coupling_vars_up)
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

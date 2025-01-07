
from typing import List, Dict, Tuple

from pyomo.environ import Var
from pyomo.core.base.var import VarData, IndexedVar

from .node import BdNode
from .cuts import Cut, OptimalityCut
from pyodec.dec.utils import CouplingData, get_nonzero_coefficients_from_model
from pyodec.core.subsolver.subsolver import SubSolver

class BdLeafNode(BdNode):
    def __init__(
            self,
            idx: int,
            sub_solver: SubSolver,
            parent: int,
            vars_up: List[VarData],
            multiplier: float = 1.0,
        ) -> None:
        super().__init__(idx, sub_solver, parent=parent, multiplier=multiplier)
        self.coupling_vars_up: List[VarData] = vars_up

        self.coupling_info: List[CouplingData] = \
            get_nonzero_coefficients_from_model(self.solver.model, self.coupling_vars_up)
        
        self.coupling_constraints = [coupling_data.constraint for coupling_data in self.coupling_info]

    def solve(self, coupling_values: List[float]) -> Cut:
        self._fix_coupling_variables(coupling_values)
        super().solve()
        if self.solver.is_optimal():
            pi = self.solver.get_dual_solution(self.coupling_constraints)
            return self._optimality_cut(pi)
        elif self.solver.is_infeasible():
            raise NotImplementedError("only implemented feasible case")
        else:
            raise ValueError("Unknown solver status")

    def _fix_coupling_variables(self, coupling_values: List[float]):
        self.fix_values = coupling_values
        self.solver.fix_variables(self.coupling_vars_up, coupling_values)

    def _optimality_cut(self, pi: List[float]) -> OptimalityCut:
        coef = [0 for _ in range(len(self.coupling_vars_up))]
        constant = self.solver.get_objective_value()
        for i, dual_var in enumerate(pi):
            coupling_data = self.coupling_info[i]
            for j, coefficients in coupling_data.coefficients.items():
                temp = dual_var * coefficients
                coef[j] += temp
                constant += temp * self.fix_values[j]
        coef = [self.multiplier * coef[i] for i in range(len(coef))]
        constant = self.multiplier * constant
        objective = self.multiplier * self.solver.get_objective_value()
        return OptimalityCut(
                coefficients=coef,
                constant=constant,
                objective_value=objective,
            )
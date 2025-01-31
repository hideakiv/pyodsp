from typing import List

from pyomo.core.base.var import VarData

from pyodec.alg.bm.cuts import Cut

from .node import DdNode
from .solver_leaf import DdSolverLeaf


class DdLeafNode(DdNode):
    def __init__(
        self,
        idx: int,
        solver: DdSolverLeaf,
        bound: float,
        parent: int,
        vars_up: List[VarData],
    ) -> None:
        super().__init__(idx, parent=parent)
        self.solver = solver
        self.coupling_vars_up: List[VarData] = vars_up
        self.bound = bound

        self.built = False

    def build(self) -> None:
        if self.built:
            return
        self.solver.build(self.coupling_vars_up)
        self.built = True

    def solve(self, coupling_values: List[float]) -> Cut:
        self.solver.fix_variables(self.coupling_vars_up, coupling_values)
        self.solver.solve()
        return self.solver.get_subgradient(self.coupling_vars_up)

    def get_bound(self) -> float:
        return self.bound

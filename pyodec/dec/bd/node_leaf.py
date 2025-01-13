from typing import List

from pyomo.core.base.var import VarData

from .node import BdNode
from .cuts import Cut

from .solver_leaf import BdSolverLeaf


class BdLeafNode(BdNode):
    def __init__(
        self,
        idx: int,
        solver: BdSolverLeaf,
        bound: float,
        parent: int,
        vars_up: List[VarData],
    ) -> None:
        super().__init__(idx, parent=parent)
        self.solver = solver
        self.coupling_vars_up: List[VarData] = vars_up
        self.bound = bound

        self.built = False

    def build(self):
        self.solver.build(self.coupling_vars_up)
        self.built = True

    def solve(self, coupling_values: List[float]) -> Cut:
        self.solver.fix_variables(self.coupling_vars_up, coupling_values)
        self.solver.solve()
        return self.solver.get_subgradient(self.coupling_vars_up)

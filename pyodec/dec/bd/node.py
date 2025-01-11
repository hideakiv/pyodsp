from pyodec.dec.node import Node
from .solver import BdSolver


class BdNode(Node):
    def __init__(
        self,
        idx: int,
        sub_solver: BdSolver,
        parent: int | None = None,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__(idx, parent=parent)
        self.multiplier = multiplier

        self.solver: BdSolver = sub_solver

    def solve(self):
        self.solver.solve()

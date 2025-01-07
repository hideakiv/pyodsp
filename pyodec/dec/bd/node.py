
from pyodec.dec.node import Node
from pyodec.core.subsolver.subsolver import SubSolver

class BdNode(Node):
    def __init__(
            self,
            idx: int,
            sub_solver: SubSolver,
            parent: int | None = None,
            multiplier: float = 1.0,
        ) -> None:
        super().__init__(idx, parent=parent)
        self.multiplier = multiplier
        
        self.solver: SubSolver = sub_solver

    def solve(self):
        self.solver.solve()




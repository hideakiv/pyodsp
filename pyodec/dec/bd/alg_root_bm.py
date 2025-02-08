from typing import List

from pyomo.core.base.var import VarData

from .alg_root import BdAlgRoot
from pyodec.solver.pyomo_solver import PyomoSolver
from pyodec.alg.bm.bm import BundleMethod
from pyodec.alg.bm.cuts import CutList


class BdAlgRootBm(BdAlgRoot):

    def __init__(self, solver: PyomoSolver, max_iteration=1000) -> None:

        self.bm = BundleMethod(solver, max_iteration)

    def get_vars(self) -> List[VarData]:
        return self.bm.solver.vars

    def build(self, subobj_bounds: List[float]) -> None:
        self.bm.build(subobj_bounds)

    def reset_iteration(self) -> None:
        self.bm.reset_iteration()

    def solve(self) -> None:
        self.bm.solve()

    def get_solution(self) -> List[float]:
        return self.bm.get_solution()

    def add_cuts(self, cuts_list: List[CutList]) -> bool:
        return self.bm.add_cuts(cuts_list)

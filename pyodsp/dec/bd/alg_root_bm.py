from typing import List
from pathlib import Path

from pyomo.core.base.var import VarData

from .alg_root import BdAlgRoot
from pyodsp.solver.pyomo_solver import PyomoSolver
from pyodsp.alg.bm.bm import BundleMethod
from pyodsp.alg.cuts import CutList


class BdAlgRootBm(BdAlgRoot):

    def __init__(self, solver: PyomoSolver, max_iteration=1000) -> None:

        self.bm = BundleMethod(solver, max_iteration)

    def get_vars(self) -> List[VarData]:
        return self.bm.solver.vars

    def build(self, subobj_bounds: List[float]) -> None:
        num_cuts = len(subobj_bounds)
        self.bm.build(num_cuts, subobj_bounds)

    def run_step(self, cuts_list: List[CutList] | None) -> List[float] | None:
        return self.bm.run_step(cuts_list)

    def reset_iteration(self) -> None:
        self.bm.reset_iteration()

    def save(self, dir: Path) -> None:
        self.bm.save(dir)

    def is_minimize(self) -> bool:
        return self.bm.solver.is_minimize()
    
    def set_logger(self, node_id: int, depth: int) -> None:
        self.bm.set_logger(node_id, depth)

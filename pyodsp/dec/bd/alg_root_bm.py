from typing import List, Tuple
from pathlib import Path
import time
import pandas as pd

from pyomo.environ import ScalarVar

from .message import BdInitDnMessage, BdDnMessage, BdFinalDnMessage, BdFinalUpMessage
from .alg_root import BdAlgRoot
from pyodsp.solver.pyomo_solver import PyomoSolver
from pyodsp.alg.bm.bm import BundleMethod
from pyodsp.alg.cuts import CutList


class BdAlgRootBm(BdAlgRoot):
    def __init__(self, solver: PyomoSolver, max_iteration=1000) -> None:
        self.bm = BundleMethod(solver, max_iteration)
        self.step_time: List[float] = []

    def get_vars(self) -> List[ScalarVar]:
        return self.bm.get_vars()

    def build(self, subobj_bounds: List[float]) -> None:
        num_cuts = len(subobj_bounds)
        self.bm.build(num_cuts, subobj_bounds)

    def run_step(self, cuts_list: List[CutList] | None) -> Tuple[int, BdDnMessage]:
        start = time.time()
        status, solution = self.bm.run_step(cuts_list)
        self.step_time.append(time.time() - start)
        return status, BdDnMessage(solution)

    def add_cuts(self, cuts_list: List[CutList]) -> None:
        self.bm.add_cuts(cuts_list)

    def reset_iteration(self) -> None:
        self.bm.reset_iteration()

    def get_final_dn_message(self, **kwargs) -> BdFinalDnMessage:
        return BdFinalDnMessage([var.value for var in self.get_vars()])

    def pass_final_up_message(self, children_obj: float) -> BdFinalUpMessage:
        obj = self.bm.get_original_objective_value() + children_obj
        return BdFinalUpMessage(obj)

    def get_num_vars(self) -> int:
        return len(self.get_vars())

    def get_init_dn_message(self, **kwargs) -> BdInitDnMessage:
        return BdInitDnMessage(self.is_minimize())

    def save(self, dir: Path) -> None:
        self.bm.save(dir)
        path = dir / "step_time.csv"
        df = pd.DataFrame(self.step_time, columns=["step_time"])
        df.to_csv(path, index=False)

    def is_minimize(self) -> bool:
        return self.bm.is_minimize()

    def set_logger(self, node_id: int, depth: int) -> None:
        self.bm.set_logger(node_id, depth)

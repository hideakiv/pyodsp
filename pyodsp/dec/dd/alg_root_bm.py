from typing import List, Dict
from pathlib import Path
import time
import pandas as pd

from pyomo.environ import ConcreteModel
from pyomo.core.base.var import VarData

from .alg_root import DdAlgRoot
from pyodsp.alg.bm.bm import BundleMethod
from pyodsp.alg.cuts import CutList
from pyodsp.alg.cuts_manager import CutInfo
from pyodsp.alg.const import BM_DUMMY_BOUND


class DdAlgRootBm(DdAlgRoot):

    def __init__(
        self,
        coupling_model: ConcreteModel,
        is_minimize: bool,
        solver_name: str,
        vars_dn: Dict[int, List[VarData]],
        max_iteration=1000,
        **kwargs
    ) -> None:
        super().__init__(coupling_model, is_minimize, solver_name, vars_dn, **kwargs)

        self.bm = BundleMethod(self.solver, max_iteration)
        self.step_time: List[float] = []

    def build(self, num_cuts: int) -> None:
        if self.is_minimize:
            dummy_bounds = [BM_DUMMY_BOUND for _ in range(num_cuts)]
        else:
            dummy_bounds = [-BM_DUMMY_BOUND for _ in range(num_cuts)]

        self.bm.build(num_cuts, dummy_bounds)

    def run_step(self, cuts_list: List[CutList] | None) -> List[float] | None:
        start = time.time()
        solution = self.bm.run_step(cuts_list)
        self.step_time.append(time.time() - start)
        return solution

    def reset_iteration(self) -> None:
        self.bm.reset_iteration()

    def get_cuts(self) -> List[List[CutInfo]]:
        return self.bm.cuts_manager.get_cuts()

    def save(self, dir: Path) -> None:
        self.bm.save(dir)
        path = dir / "step_time.csv"
        df = pd.DataFrame(self.step_time, columns=["step_time"])
        df.to_csv(path, index=False)
    
    def set_logger(self, node_id: int, depth: int) -> None:
        self.bm.set_logger(node_id, depth)

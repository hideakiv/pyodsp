from typing import List, Dict, Tuple
from pathlib import Path
import time
import pandas as pd

from pyomo.environ import ConcreteModel, ScalarVar

from .alg_root import DdAlgRoot
from .message import DdDnMessage
from pyodsp.alg.pbm.pbm import ProximalBundleMethod
from pyodsp.alg.cuts import CutList
from pyodsp.alg.cuts_manager import CutInfo


class DdAlgRootPbm(DdAlgRoot):
    def __init__(
        self,
        coupling_model: ConcreteModel,
        is_minimize: bool,
        solver_name: str,
        vars_dn: Dict[int, List[ScalarVar]],
        max_iteration=1000,
        **kwargs,
    ) -> None:
        super().__init__(coupling_model, is_minimize, solver_name, vars_dn, **kwargs)

        self.pbm = ProximalBundleMethod(self.solver, max_iteration)
        self.step_time: List[float] = []

    def build(self, bounds: List[float | None]) -> None:
        num_cuts = len(bounds)
        self.pbm.set_init_solution([0.0 for _ in range(num_cuts)])
        self.pbm.build(num_cuts)

    def run_step(self, cuts_list: List[CutList] | None) -> Tuple[int, DdDnMessage]:
        start = time.time()
        status, solution = self.pbm.run_step(cuts_list)
        self.step_time.append(time.time() - start)
        return status, DdDnMessage(solution)

    def reset_iteration(self) -> None:
        self.pbm.reset_iteration()

    def get_dn_message(self) -> DdDnMessage:
        return DdDnMessage([var.value for var in self.pbm.solver.vars])

    def get_num_vars(self) -> int:
        return len(self.pbm.solver.vars)

    def add_cuts(self, cuts_list: List[CutList]) -> None:
        self.pbm.add_cuts(cuts_list)

    def get_cuts(self) -> List[List[CutInfo]]:
        return self.pbm.cuts_manager.get_cuts()

    def save(self, dir: Path) -> None:
        self.pbm.save(dir)
        path = dir / "step_time.csv"
        df = pd.DataFrame(self.step_time, columns=["step_time"])
        df.to_csv(path, index=False)

    def set_logger(self, node_id: int, depth: int) -> None:
        self.pbm.set_logger(node_id, depth)

from typing import List, Dict, Tuple
from pathlib import Path
import time
import pandas as pd

from pyomo.environ import ConcreteModel, ScalarVar

from .message import DdDnMessage, DdFinalDnMessage
from .alg_root import DdAlgRoot
from pyodsp.alg.bm.bm import BundleMethod
from pyodsp.alg.cuts import CutList
from pyodsp.alg.cuts_manager import CutInfo
from pyodsp.alg.params import BM_DUMMY_BOUND
from pyodsp.solver.pyomo_solver import SolverConfig


class DdAlgRootBm(DdAlgRoot):
    def __init__(
        self,
        coupling_model: ConcreteModel,
        is_minimize: bool,
        solver_config: SolverConfig,
        final_solver_config: SolverConfig | None,
        vars_dn: Dict[int, List[ScalarVar]],
        max_iteration=1000,
    ) -> None:
        super().__init__(
            coupling_model, is_minimize, solver_config, final_solver_config, vars_dn
        )

        self.bm = BundleMethod(self.solver, max_iteration)
        self.step_time: List[float] = []

    def build(self, bounds: List[float | None]) -> None:
        num_cuts = len(bounds)
        if self.is_minimize():
            dummy_bounds = [BM_DUMMY_BOUND for _ in range(num_cuts)]
        else:
            dummy_bounds = [-BM_DUMMY_BOUND for _ in range(num_cuts)]

        self.bm.build(num_cuts, dummy_bounds)

    def run_step(self, cuts_list: List[CutList] | None) -> Tuple[int, DdDnMessage]:
        start = time.time()
        status, solution = self.bm.run_step(cuts_list)
        self.step_time.append(time.time() - start)
        return status, DdDnMessage(solution)

    def reset_iteration(self) -> None:
        self.bm.reset_iteration()

    def get_final_dn_message(self, **kwargs) -> DdFinalDnMessage:
        if self.final_solver_config is None:
            return DdFinalDnMessage(None)
        super().get_final_dn_message(**kwargs)
        node_id = kwargs["node_id"]
        return self.final_solutions[node_id]

    def get_num_vars(self) -> int:
        return self.bm.get_num_vars()

    def add_cuts(self, cuts_list: List[CutList]) -> None:
        self.bm.add_cuts(cuts_list)

    def get_cuts(self) -> List[List[CutInfo]]:
        return self.bm.get_cuts()

    def save(self, dir: Path) -> None:
        self.bm.save(dir)
        path = dir / "step_time.csv"
        df = pd.DataFrame(self.step_time, columns=["step_time"])
        df.to_csv(path, index=False)

    def set_logger(self, node_id: int, depth: int) -> None:
        self.bm.set_logger(node_id, depth)

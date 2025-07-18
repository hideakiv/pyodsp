from typing import List, Dict, Tuple
from pathlib import Path
import time
import pandas as pd

from pyomo.environ import ConcreteModel, ScalarVar

from .alg_root import DdAlgRoot
from .message import DdDnMessage, DdFinalDnMessage
from .mip_heuristic_root import IMipHeuristicRoot
from pyodsp.alg.pbm.pbm import ProximalBundleMethod
from pyodsp.alg.cuts import CutList
from pyodsp.alg.cuts_manager import CutInfo
from pyodsp.solver.pyomo_solver import SolverConfig


class DdAlgRootPbm(DdAlgRoot):
    def __init__(
        self,
        coupling_model: ConcreteModel,
        is_minimize: bool,
        solver_config: SolverConfig,
        vars_dn: Dict[int, List[ScalarVar]],
        heuristic: IMipHeuristicRoot | None = None,
        max_iteration=1000,
    ) -> None:
        super().__init__(coupling_model, is_minimize, solver_config, vars_dn, heuristic)

        self.pbm = ProximalBundleMethod(self.solver, max_iteration)
        self.step_time: List[float] = []

    def build(self, bounds: List[float | None]) -> None:
        num_cuts = len(bounds)
        self.pbm.set_init_solution([0.0 for _ in range(self.num_constrs)])
        self.pbm.build(num_cuts)

    def run_step(self, cuts_list: List[CutList] | None) -> Tuple[int, DdDnMessage]:
        start = time.time()
        status, solution = self.pbm.run_step(cuts_list)
        self.step_time.append(time.time() - start)
        return status, DdDnMessage(solution)

    def reset_iteration(self) -> None:
        self.pbm.reset_iteration()

    def get_final_dn_message(self, **kwargs) -> DdFinalDnMessage:
        if self.heuristic is None:
            return DdFinalDnMessage(None)
        super().get_final_dn_message(**kwargs)
        node_id = kwargs["node_id"]
        return self.final_solutions[node_id]

    def get_num_vars(self) -> int:
        return self.pbm.get_num_vars()

    def add_cuts(self, cuts_list: List[CutList]) -> None:
        self.pbm.add_cuts(cuts_list)

    def get_cuts(self) -> List[List[CutInfo]]:
        return self.pbm.get_cuts()

    def save(self, dir: Path) -> None:
        self.pbm.save(dir)
        path = dir / "step_time.csv"
        df = pd.DataFrame(self.step_time, columns=["step_time"])
        df.to_csv(path, index=False)

    def set_logger(self, node_id: int, depth: int) -> None:
        self.pbm.set_logger(node_id, depth)

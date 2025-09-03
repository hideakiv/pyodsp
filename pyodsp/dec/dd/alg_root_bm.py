from typing import List, Dict, Tuple
from pathlib import Path
import time
import pandas as pd
import logging

from pyomo.environ import ConcreteModel, ScalarVar, Objective, Var

from ..node._alg import IAlgRoot
from .message import DdDnMessage, DdFinalDnMessage, DdInitDnMessage, DdFinalUpMessage
from .master_creator import MasterCreator
from .mip_heuristic_root import IMipHeuristicRoot
from pyodsp.alg.bm.bm import BundleMethod
from pyodsp.alg.bm.pbm import ProximalBundleMethod
from pyodsp.alg.bm.cuts import CutList
from pyodsp.alg.bm.cuts_manager import CutInfo
from pyodsp.alg.params import BM_DUMMY_BOUND
from pyodsp.solver.pyomo_solver import SolverConfig


class DdAlgRootBm(IAlgRoot):
    def __init__(
        self,
        coupling_model: ConcreteModel,
        is_minimize: bool,
        solver_config: SolverConfig,
        vars_dn: Dict[int, List[ScalarVar]],
        heuristic: IMipHeuristicRoot | None = None,
        max_iteration=1000,
        mode: str | None = None,
    ) -> None:
        self.coupling_model = coupling_model
        self.vars_dn = vars_dn
        self._init_check()
        mc = MasterCreator(coupling_model, is_minimize, solver_config, vars_dn)
        self.solver = mc.create()
        self.lagrangian_data = mc.lagrangian_data
        self.num_constrs = mc.num_constrs
        self._is_minimize = is_minimize
        self.heuristic = heuristic

        self.is_finalized = False

        self.mode = mode
        if mode is None:
            self.bm = BundleMethod(self.solver, max_iteration)
        elif mode == "proximal":
            self.bm = ProximalBundleMethod(self.solver, max_iteration)
        else:
            raise ValueError(f"Invalid mode {mode}")
        self.step_time: List[float] = []

    def get_vars_dn(self) -> Dict[int, List[ScalarVar]]:
        return self.vars_dn

    def _init_check(self) -> None:
        for obj in self.coupling_model.component_objects(Objective, active=True):
            # There should not be any objective
            raise ValueError("Objective should not be defined in coupling_model")

        # Check that vars_dn is properly specified
        varname_list = []
        for var in self.coupling_model.component_objects(ctype=Var):
            if isinstance(var, ScalarVar):
                varname_list.append(var.name)
            else:
                for index in var:
                    varname_list.append(var[index].name)

        for varlist in self.vars_dn.values():
            for var in varlist:
                if var.name in varname_list:
                    varname_list.pop(varname_list.index(var.name))
                else:
                    raise ValueError(
                        f"Variable {var.name} does not exist in varname_list"
                    )

        if len(varname_list) > 0:
            raise ValueError(f"Variables {varname_list} not coupled")

    def is_minimize(self) -> bool:
        return self._is_minimize

    def get_init_dn_message(self, **kwargs) -> DdInitDnMessage:
        child_id = kwargs["child_id"]
        message = DdInitDnMessage(
            self.lagrangian_data.matrix[child_id], self.is_minimize()
        )
        return message

    def get_coupling_model(self) -> ConcreteModel:
        return self.coupling_model

    def build(self, bounds: List[float | None]) -> None:
        num_cuts = len(bounds)
        if self.mode is None:
            assert type(self.bm) is BundleMethod
            if self.is_minimize():
                dummy_bounds = [BM_DUMMY_BOUND for _ in range(num_cuts)]
            else:
                dummy_bounds = [-BM_DUMMY_BOUND for _ in range(num_cuts)]

            self.bm.build(num_cuts, dummy_bounds)
        elif self.mode == "proximal":
            assert type(self.bm) is ProximalBundleMethod
            self.bm.set_init_solution([0.0 for _ in range(self.num_constrs)])
            self.bm.build(num_cuts)

    def run_step(self, cuts_list: List[CutList] | None) -> Tuple[int, DdDnMessage]:
        start = time.time()
        status, solution = self.bm.run_step(cuts_list)
        self.step_time.append(time.time() - start)
        return status, DdDnMessage(solution)

    def reset_iteration(self) -> None:
        self.bm.reset_iteration()

    def get_final_dn_message(self, **kwargs) -> DdFinalDnMessage:
        if self.heuristic is None:
            return DdFinalDnMessage(None)
        if not self.is_finalized:
            groups = kwargs["groups"]
            self.heuristic.build(
                groups=groups,
                coupling_model=self.coupling_model,
                cuts=self.get_cuts(),
                vars_dn=self.get_vars_dn(),
                is_minimize=self.is_minimize(),
            )
            self.final_solutions = self.heuristic.run_init()
            self.is_finalized = True
        node_id = kwargs["node_id"]
        return self.final_solutions[node_id]

    def pass_final_up_message(self, children_obj: float | None) -> DdFinalUpMessage:
        return DdFinalUpMessage(children_obj)

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

    def set_logger(self, node_id: int, depth: int, level: int = logging.INFO) -> None:
        self.bm.set_logger(node_id, depth, level)

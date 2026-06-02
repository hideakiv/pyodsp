from typing import List, Tuple
from pathlib import Path
import time
import pandas as pd
import logging

from pyomo.environ import ScalarVar

from .message import (
    BdScInitDnMessage,
    BdScUpMessage,
    BdScDnMessage,
    BdScFinalDnMessage,
    BdScFinalUpMessage,
)
from ..node._alg import IAlgRoot
from pyodsp.solver.pyomo_solver import PyomoSolver
from pyodsp.alg.bm.bm import BundleMethod
from pyodsp.dec.node._message import NodeIdx
from pyodsp.dec.node.cut_aggregator import CutAggregator


class BdScAlgRootBm(IAlgRoot):
    def __init__(self, solver: PyomoSolver, max_iteration=1000) -> None:
        self.bm = BundleMethod(solver, max_iteration)
        self.step_time: List[float] = []

    def get_vars(self) -> List[ScalarVar]:
        return self.bm.get_vars()

    def build(
        self,
        groups: list[list[NodeIdx]],
        children_multipliers: dict[NodeIdx, float],
        children_bounds: dict[NodeIdx, float],
    ) -> None:
        num_cuts = len(groups)
        self.groups = groups
        self.children_multipliers = children_multipliers
        self.cut_aggregator = CutAggregator(self.groups, self.children_multipliers)
        subobj_bounds: List[float | None] = []
        for group in self.groups:
            bound = 0.0
            for member in group:
                if member not in children_bounds:
                    bound = None
                    break
                bound += self.children_multipliers[member] * children_bounds[member]
            subobj_bounds.append(bound)
        self.bm.build(num_cuts, subobj_bounds)
        self.rho = None
        self.solution = None
        self.objective = None

    def run_step(
        self, up_messages: dict[NodeIdx, BdScUpMessage] | None
    ) -> Tuple[int, BdScDnMessage]:
        if up_messages is None:
            cuts_list = None
            start = time.time()
            status, solution, objective = self.bm.run_step(cuts_list)
            self.step_time.append(time.time() - start)
            self.rho = sum(self.bm.get_theta_value())
            self.solution = solution
            self.objective = objective
        else:
            c_average = 0.0
            tau_average = 0.0
            for node_idx, message in up_messages.items():
                c_average += message.get_c() * self.children_multipliers[node_idx]
                tau_average += message.get_tau() * self.children_multipliers[node_idx]
            if c_average <= 1e-6:
                cuts_list = self.cut_aggregator.get_aggregate_cuts(up_messages)
                for cuts in cuts_list:
                    for cut in cuts:
                        for var_id, coeff in cut.coeffs().items():
                            cut.coeffs[var_id] = coeff / (1 + tau_average)

                start = time.time()
                status, solution, objective = self.bm.run_step(cuts_list)
                self.step_time.append(time.time() - start)
                self.rho = sum(self.bm.get_theta_value())
                self.solution = solution
                self.objective = objective
            else:
                cuts_list = None
                assert self.rho is not None
                self.rho = self.rho + c_average / (1 + tau_average)

        return status, BdScDnMessage(self.solution, self.rho, cuts_list, self.objective)

    def add_cuts(self, up_messages: dict[NodeIdx, BdScUpMessage]) -> None:
        cuts_list = self.cut_aggregator.get_aggregate_cuts(up_messages)
        self.bm.add_cuts(cuts_list)

    def reset_iteration(self) -> None:
        self.bm.reset_iteration()

    def get_final_dn_message(self, **kwargs) -> BdScFinalDnMessage:
        return BdScFinalDnMessage([var.value for var in self.get_vars()])

    def pass_final_up_message(
        self, messages: dict[NodeIdx, BdScFinalUpMessage]
    ) -> BdScFinalUpMessage:
        obj = self.bm.get_original_objective_value()
        for message in messages.values():
            child_obj = message.get_objective()
            if child_obj is None:
                return BdScFinalUpMessage(None)
            obj += child_obj
        return BdScFinalUpMessage(obj)

    def get_num_vars(self) -> int:
        return len(self.get_vars())

    def get_init_dn_message(self, **kwargs) -> BdScInitDnMessage:
        return BdScInitDnMessage(self.is_minimize())

    def save(self, dir: Path) -> None:
        self.bm.save(dir)
        path = dir / "step_time.csv"
        df = pd.DataFrame(self.step_time, columns=["step_time"])
        df.to_csv(path, index=False)

    def is_minimize(self) -> bool:
        return self.bm.is_minimize()

    def set_logger(self, node_id: int, depth: int, level: int = logging.INFO) -> None:
        self.bm.set_logger(node_id, depth, level)

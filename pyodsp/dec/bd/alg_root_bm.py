from typing import List, Tuple
from pathlib import Path
import time
import pandas as pd
import logging

from pyomo.environ import ScalarVar

from .message import (
    BdInitDnMessage,
    BdUpMessage,
    BdDnMessage,
    BdFinalDnMessage,
    BdFinalUpMessage,
)
from ..node._alg import IAlgRoot
from pyodsp.solver.pyomo_solver import PyomoSolver
from pyodsp.alg.bm.bm import BundleMethod
from pyodsp.dec.node._message import NodeIdx
from pyodsp.dec.node.cut_aggregator import CutAggregator


class BdAlgRootBm(IAlgRoot):
    def __init__(self, solver: PyomoSolver, max_iteration=1000) -> None:
        if not solver.is_minimize():
            raise ValueError(
                "Benders decomposition only accepts minimize problems; "
                "negate the objective (see pyomo_utils.negate_objective_sense) "
                "before constructing the solver."
            )
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

    def run_step(
        self, up_messages: dict[NodeIdx, BdUpMessage] | None
    ) -> Tuple[int, BdDnMessage]:
        if up_messages is None:
            cuts_list = None
        else:
            cuts_list = self.cut_aggregator.get_aggregate_cuts(up_messages)
        start = time.time()
        status, solution, objective = self.bm.run_step(cuts_list)
        self.step_time.append(time.time() - start)
        return status, BdDnMessage(solution, objective)

    def add_cuts(self, up_messages: dict[NodeIdx, BdUpMessage]) -> None:
        cuts_list = self.cut_aggregator.get_aggregate_cuts(up_messages)
        self.bm.add_cuts(cuts_list)

    def reset_iteration(self) -> None:
        self.bm.reset_iteration()

    def get_final_dn_message(self, **kwargs) -> BdFinalDnMessage:
        return BdFinalDnMessage([var.value for var in self.get_vars()])

    def pass_final_up_message(
        self, messages: dict[NodeIdx, BdFinalUpMessage]
    ) -> BdFinalUpMessage:
        obj = self.bm.get_original_objective_value()
        for message in messages.values():
            child_obj = message.get_objective()
            if child_obj is None:
                return BdFinalUpMessage(None)
            obj += child_obj
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

    def set_logger(self, node_id: int, depth: int, level: int = logging.INFO) -> None:
        self.bm.set_logger(node_id, depth, level)

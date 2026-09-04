from typing import List, Tuple
from copy import deepcopy
from pathlib import Path
import time
import logging

import pandas as pd

from pyomo.environ import Var, ScalarVar, Reals, RangeSet

from pyodsp.solver.pyomo_solver import PyomoSolver

from .cuts import CutList
from .cuts_manager import CutInfo
from .logger import BmLogger
from .cp import CuttingPlaneMethod
from ..params import BM_ABS_TOLERANCE, BM_REL_TOLERANCE, BM_PURGE_FREQ, BM_TIME_LIMIT
from ..const import *


class BundleMethod:
    def __init__(
        self,
        solver: PyomoSolver,
        max_iteration=1000,
        force: bool = False,
        purgeable: bool = True,
        risk=None,
    ) -> None:
        self.cpm = CuttingPlaneMethod(solver, force, purgeable, risk=risk)

        self.max_iteration = max_iteration
        self.iteration = 0

        self.obj_bound: List[float | None] = []
        self.obj_val: List[float | None] = []

        self.status: int = STATUS_NOT_FINISHED
        self.start_time = time.time()

    def set_logger(
        self, node_id: int | str, depth: int, level: int = logging.INFO
    ) -> None:
        method = "Bundle Method"
        self.logger = BmLogger(method, node_id, depth, level)

    def build(
        self,
        num_cuts: int,
        subobj_bounds: List[float] | None,
        group_probabilities: List[float] | None = None,
    ) -> None:
        """
        Args:
            num_cuts: One theta per cut group.
            subobj_bounds: A floor under each theta, or None entries where
                none is known.
            group_probabilities: Each group's probability. Only a risk
                measure needs them — it prices the spread across groups,
                where the expectation only needs their sum.
        """
        self.num_cuts = num_cuts
        assert subobj_bounds is not None
        self.subobj_bounds = subobj_bounds
        assert self.num_cuts == len(subobj_bounds)
        self._update_objective(subobj_bounds, group_probabilities)
        self.cpm.build(self.num_cuts)

        self.logger.log_initialization(
            tolerance=BM_ABS_TOLERANCE, max_iteration=self.max_iteration
        )

    def run_step(
        self, cuts_list: List[CutList] | None
    ) -> Tuple[int, List[float] | None, float]:
        if cuts_list is not None:
            no_cuts, feasible, obj_val = self.add_cuts(cuts_list)
            if feasible:
                self.obj_val.append(obj_val)
            else:
                self.obj_val.append(None)
        else:
            self.obj_val.append(None)

        self._increment()
        self.cpm.solve()
        if self.cpm.is_infeasible():
            self.status = STATUS_INFEASIBLE
            self.logger.log_infeasible()
            return self.status, None, 0.0

        current_obj = self.cpm.get_relaxed_objective()
        self.obj_bound.append(current_obj)

        self._log()
        if self._termination_check():
            self.logger.log_completion(self.iteration, self.obj_bound[-1])

        parent_objective = self.cpm.get_parent_objective_value()
        original_objective = self.cpm.get_original_objective_value()
        sample_objective = parent_objective + original_objective

        return (
            self.status,
            self.cpm.get_current_solution(),
            sample_objective,
        )

    def reset_iteration(self, i=0) -> None:
        self.iteration = i
        self.status = STATUS_NOT_FINISHED
        self.start_time = time.time()

    def is_minimize(self) -> bool:
        return self.cpm.is_minimize()

    def get_cuts(self) -> List[List[CutInfo]]:
        return self.cpm.get_cuts()

    def get_vars(self) -> List[ScalarVar]:
        return self.cpm.get_vars()

    def get_num_vars(self) -> int:
        return len(self.get_vars())

    def get_original_objective_value(self) -> float:
        return self.cpm.get_original_objective_value()

    def get_objective_value(self) -> float:
        return self.cpm.get_objective_value()

    def _log(self) -> None:
        if self.is_minimize():
            lb = self.obj_bound[-1]
            ub = self.obj_val[-1]
        else:
            lb = self.obj_val[-1]
            ub = self.obj_bound[-1]
        self.logger.log_iteration(
            iteration=self.iteration,
            lb=lb,
            ub=ub,
            num_cuts=self.cpm.get_num_cuts(),
            elapsed=time.time() - self.start_time,
        )
        self.logger.log_solution(self.cpm.get_current_solution())

    def _termination_check(self) -> bool:
        if self.iteration >= self.max_iteration:
            self.status = STATUS_MAX_ITERATION
            self.logger.log_status_max_iter()
            return True

        if time.time() - self.start_time > BM_TIME_LIMIT:
            self.status = STATUS_TIME_LIMIT
            self.logger.log_status_time_limit()
            return True

        if (
            len(self.obj_val) == 0
            or self.obj_val[-1] is None
            or self.obj_bound[-1] is None
        ):
            return False

        for i in range(self.num_cuts):
            bound_gap = abs(self.cpm.get_theta_value(i) - self.subobj_bounds[i])
            if bound_gap < BM_ABS_TOLERANCE:
                return False
        if abs(self.obj_val[-1]) < BM_ABS_TOLERANCE:
            gap = abs(self.obj_bound[-1] - self.obj_val[-1]) / BM_ABS_TOLERANCE
        else:
            gap = abs(self.obj_bound[-1] - self.obj_val[-1]) / abs(self.obj_val[-1])

        if gap < BM_REL_TOLERANCE:
            self.status = STATUS_OPTIMAL
            self.logger.log_status_optimal()
            return True

        return False

    def in_user_units(self, values: List[float | None]) -> List[float | None]:
        return self.cpm.in_user_units(values)

    def get_objective_bound(self) -> float | None:
        """The latest bound on the optimum, in the caller's units."""
        if not self.obj_bound:
            return None
        return self.cpm.to_user_units(self.obj_bound[-1])

    def save(self, dir: Path) -> None:
        path = dir / "bm.csv"
        df = pd.DataFrame(
            {
                "obj_bound": self.in_user_units(self.obj_bound),
                "obj_val": self.in_user_units(self.obj_val),
            }
        )
        df.to_csv(path)
        self.cpm.save(dir)

    def add_cuts(self, cuts_list: List[CutList]) -> Tuple[bool, bool, float | None]:
        return self.cpm.add_cuts(cuts_list)

    def _increment(self) -> None:
        self.iteration += 1
        self.cpm.increment_cuts()
        if self.iteration % BM_PURGE_FREQ == 0:
            self.cpm.purge_cuts()

    def get_cut_list(self) -> List[CutList]:
        """A snapshot of currently active cuts as CutList objects (dropping
        the constraint/age/trial_point bookkeeping), suitable for relaying
        to another BundleMethod via replace_cuts.

        The cuts are copied, so a recipient sharing this process (see
        BdScAlgLeafPyomo) cannot reach back into this master's own cuts —
        callers do rescale cuts in place (see BdScAlgRootBm.run_step), and
        a snapshot that aliased them would not stay a snapshot.
        """
        return [CutList([deepcopy(c.cut) for c in group]) for group in self.get_cuts()]

    def replace_cuts(self, cuts_list: List[CutList]) -> None:
        """Replace every currently active cut with exactly the given set,
        bypassing this BundleMethod's own add/dominance decisions.

        Intended for a force=True replica (see BdAlgRootBm's purgeable
        flag) mirroring another BundleMethod's cuts precisely (via
        get_cut_list). Incrementally replaying add_cuts calls without the
        same interleaved solves the source of truth performed would make
        an independently-evaluated dominance check diverge from it —
        replacing wholesale avoids re-deciding anything.
        """
        all_names = [c.constraint.name for group in self.get_cuts() for c in group]
        self.cpm.eliminate_cuts(all_names)
        self.add_cuts(cuts_list)

    def restore_cuts(self, dir: Path) -> None:
        """Reinstate the cuts a previous run saved (see save) into this
        freshly built master, replacing whatever it currently holds.

        This is what makes the saved value function reusable: rebuild the
        model exactly as the run built it, restore, and the master prices
        the future the same way it did at the end of the run — without
        depending on the state it was last solved at.
        """
        self.replace_cuts(self.cpm.load_cuts(dir))

    def get_solver(self) -> PyomoSolver:
        return self.cpm.get_solver()

    def _update_objective(self, subobj_bounds: List[float], group_probabilities=None):
        sign = self.cpm.get_sign()

        def theta_bounds(model, i):
            bound = subobj_bounds[i]
            return (None, None) if bound is None else (sign * bound, None)

        solver = self.cpm.get_solver()

        solver.model._theta = Var(
            RangeSet(0, self.num_cuts - 1), domain=Reals, bounds=theta_bounds
        )

        self.cpm.build_theta_objective(solver.model._theta, group_probabilities)

    def get_theta_value(self) -> list[float]:
        return [self.cpm.get_theta_value(i) for i in range(self.num_cuts)]

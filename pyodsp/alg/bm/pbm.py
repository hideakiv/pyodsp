from typing import List, Tuple
from pathlib import Path
import time
import logging

import numpy as np
import pandas as pd
from pyomo.environ import Var, ScalarVar, Reals, RangeSet

from pyodsp.alg.bm.cuts import CutList, OptimalityCut, FeasibilityCut

from .logger import BmLogger
from .cp import CuttingPlaneMethod
from .cuts_manager import CutInfo
from ..params import (
    BM_ABS_TOLERANCE,
    BM_REL_TOLERANCE,
    BM_PURGE_FREQ,
    BM_TIME_LIMIT,
    PBM_ML,
    PBM_MR,
    PBM_U_MIN,
    PBM_E_S,
)
from ..const import *
from pyodsp.solver.pyomo_solver import PyomoSolver
from pyodsp.solver.pyomo_utils import (
    add_quad_terms_to_objective,
    update_quad_terms_in_objective,
)

"""
Kiwiel, K. C. (1990). 
Proximity control in bundle methods for convex nondifferentiable minimization. 
Mathematical programming, 46(1), 105-122.

"""


class ProximalBundleMethod:
    def __init__(self, solver: PyomoSolver, max_iteration=1000, penalty=1.0) -> None:
        self.cpm = CuttingPlaneMethod(solver)

        self.max_iteration = max_iteration
        self.iteration = 0

        self.obj_bound: List[float | None] = []
        self.obj_val: List[float | None] = []

        self.status: int = STATUS_NOT_FINISHED
        self.start_time = time.time()

        self.penalty = penalty
        self.center_val = []
        self.iter_since_update = 0
        self.e_v = np.inf

    def set_logger(self, node_id: int, depth: int, level: int = logging.INFO) -> None:
        method = "Proximal Bundle Method"
        self.logger = BmLogger(method, node_id, depth, level)

    def set_init_solution(self, solution: List[float]) -> None:
        self.center = solution

    def reset_iteration(self, i=0) -> None:
        self.iteration = i
        self.status = STATUS_NOT_FINISHED
        self.start_time = time.time()

    def is_minimize(self) -> bool:
        return self.cpm.is_minimize()

    def build(self, num_cuts: int, subobj_bounds: List[float] | None = None) -> None:
        self.num_cuts = num_cuts
        self.subobj_bounds = subobj_bounds
        if subobj_bounds is not None:
            assert self.num_cuts == len(subobj_bounds)
        self._update_objective(subobj_bounds)
        self.cpm.build(self.num_cuts)

        self.logger.log_initialization(
            tolerance=BM_ABS_TOLERANCE, max_iteration=self.max_iteration
        )

    def run_step(
        self, cuts_list: List[CutList] | None
    ) -> Tuple[int, List[float] | None]:
        if cuts_list is not None:
            no_cuts, feasible, obj_val = self.add_cuts(cuts_list)
            if feasible:
                self.obj_val.append(obj_val)
            else:
                self.obj_val.append(None)

            if no_cuts or self._improved():
                # serious step
                self.logger.log_debug("Serious Step")
                self._serious_step_penalty_update()
                self.center_val.append(self.obj_val[-1])
            elif len(self.center_val) == 0 and obj_val is not None:
                self.center_val.append(self.obj_val[-1])
            else:
                self.logger.log_debug("Null Step")
                self._null_step_penalty_update(cuts_list)
                self.center_val.append(self.center_val[-1])
        else:
            self.obj_val.append(None)
            self.center_val.append(None)

        self._increment()

        self.cpm.solve()
        if self.cpm.is_infeasible():
            self.status = STATUS_INFEASIBLE
            self.logger.log_infeasible()
            return self.status, None

        current_obj = self.cpm.get_relaxed_objective()
        self.obj_bound.append(current_obj)

        self._log()

        if self._termination_check():
            self.logger.log_completion(self.iteration, self.obj_bound[-1])

        return self.status, self.cpm.get_current_solution()

    def _log(self) -> None:
        if self.is_minimize():
            lb = self.obj_bound[-1]
            ub = self.obj_val[-1]
        else:
            lb = self.obj_val[-1]
            ub = self.obj_bound[-1]
        numcuts = self.cpm.get_num_cuts()
        elapsed = time.time() - self.start_time
        if lb is None:
            lb = "-"
        else:
            lb = f"{lb:.4f}"
        cb = self.center_val[-1]
        if cb is None:
            cb = "-"
        else:
            cb = f"{cb:.4f}"
        if ub is None:
            ub = "-"
        else:
            ub = f"{ub:.4f}"
        self.logger.log_info(
            f"Iteration: {self.iteration}\tLB: {lb}\t CB: {cb}\t UB: {ub}\t NumCuts: {numcuts}\t u: {self.penalty}\t Elapsed: {elapsed:.2f}"
        )
        self.logger.log_debug(f"\tsolution: {self.cpm.get_current_solution()}")

    def get_cuts(self) -> List[List[CutInfo]]:
        return self.cpm.get_cuts()

    def get_vars(self) -> List[ScalarVar]:
        return self.cpm.get_vars()

    def get_num_vars(self) -> int:
        return len(self.get_vars())

    def _termination_check(self) -> bool:
        if self.iteration >= self.max_iteration:
            self.status = STATUS_MAX_ITERATION
            self.logger.log_status_max_iter()
            return True

        if time.time() - self.start_time > BM_TIME_LIMIT:
            self.status = STATUS_TIME_LIMIT
            self.logger.log_status_time_limit()
            return True

        if len(self.center_val) == 0 or self.center_val[-1] is None:
            return False

        if self.subobj_bounds is not None:
            for i in range(self.num_cuts):
                bound_gap = abs(self.cpm.get_theta_value(i) - self.subobj_bounds[i])
                if bound_gap < BM_ABS_TOLERANCE:
                    return False

        obj_val = self.obj_val[-1]
        center_val = self.center_val[-1]
        if obj_val is None or center_val is None:
            return False
        approx_val = self.cpm.get_relaxed_objective()
        predicted_diff = approx_val - center_val

        if abs(predicted_diff) <= PBM_E_S * (1 + abs(center_val)):
            if len(self.obj_bound) > len(self.center_val):
                self.center_val.append(self.center_val[-1])
            if len(self.obj_bound) > len(self.obj_val):
                self.obj_val.append(self.obj_val[-1])
            self.status = STATUS_OPTIMAL
            self.logger.log_status_optimal()
            return True

        return False

    def save(self, dir: Path) -> None:
        path = dir / "pbm.csv"
        df = pd.DataFrame(
            {
                "obj_bound": self.obj_bound,
                "center_val": self.center_val,
                "obj_val": self.obj_val,
            }
        )
        df.to_csv(path)
        self.cpm.save(dir)

    def add_cuts(self, cuts_list: List[CutList]) -> Tuple[bool, bool, float]:
        return self.cpm.add_cuts(cuts_list)

    def _increment(self) -> None:
        self.iteration += 1
        self.cpm.increment_cuts()
        if self.iteration % BM_PURGE_FREQ == 0:
            self.cpm.purge_cuts()

    def _improved(self) -> bool:
        if len(self.center_val) == 0 or self.center_val[-1] is None:
            return False
        obj_val = self.obj_val[-1]
        if obj_val is None:
            return False
        center_val = self.center_val[-1]
        approx_val = self.cpm.get_relaxed_objective()
        predicted_diff = approx_val - center_val

        if self.is_minimize():
            # Minimization
            return obj_val <= center_val + PBM_ML * predicted_diff
        else:
            # Maximization
            return obj_val >= center_val + PBM_ML * predicted_diff

    def _update_objective(self, subobj_bounds: List[float] | None):
        def theta_bounds(model, i):
            if subobj_bounds is None:
                return (None, None)
            if self.is_minimize():
                # Minimization
                return (subobj_bounds[i], None)
            else:
                # Maximization
                return (None, subobj_bounds[i])

        solver = self.cpm.get_solver()

        solver.model._theta = Var(
            RangeSet(0, self.num_cuts - 1), domain=Reals, bounds=theta_bounds
        )

        add_quad_terms_to_objective(
            solver,
            solver.model._theta,
            solver.vars,
            self.center,
            self.penalty,
        )

    def _update_center(self, center: List[float]) -> None:
        self.center = center
        solver = self.cpm.get_solver()
        update_quad_terms_in_objective(solver, solver.vars, self.center, self.penalty)

    def _serious_step_penalty_update(self):
        obj_val = self.obj_val[-1]
        center_val = self.center_val[-1]
        approx_val = self.cpm.get_relaxed_objective()
        predicted_diff = approx_val - center_val
        if self.is_minimize():
            penalty_too_large = obj_val <= center_val + PBM_MR * predicted_diff
        else:
            penalty_too_large = obj_val >= center_val + PBM_MR * predicted_diff

        if penalty_too_large and self.iter_since_update > 0:
            u = 2 * self.penalty * (1 - (obj_val - center_val) / predicted_diff)
        elif self.iter_since_update > 3:
            u = self.penalty / 2
        else:
            u = self.penalty

        newu = max(u, self.penalty / 10, PBM_U_MIN)
        self.e_v = max(self.e_v, abs(2 * predicted_diff))
        self.iter_since_update = max(self.iter_since_update + 1, 1)
        if abs(newu - self.penalty) > BM_ABS_TOLERANCE:
            self.penalty = newu
            self.iter_since_update = 1
        self._update_center(self.cpm.get_current_solution())

    def _get_alpha(self, cuts_list: List[CutList]) -> float | None:
        obj_val = self.obj_val[-1]
        center_val = self.center_val[-1]
        if obj_val is None or center_val is None:
            return None

        d = np.array(self.cpm.get_current_solution()) - np.array(self.center)
        grad = [0.0] * len(d)
        for idx, cuts in enumerate(cuts_list):
            for cut in cuts:
                if isinstance(cut, OptimalityCut):
                    for i, coeff in cut.coeffs.items():
                        grad[i] += coeff
                elif isinstance(cut, FeasibilityCut):
                    return None
        return center_val - obj_val + np.inner(grad, d)

    def _null_step_penalty_update(self, cuts_list: List[CutList]):
        obj_val = self.obj_val[-1]
        center_val = self.center_val[-1]
        approx_val = self.cpm.get_relaxed_objective()
        predicted_diff = approx_val - center_val
        d = np.array(self.cpm.get_current_solution()) - np.array(self.center)

        p = np.linalg.norm(-self.penalty * d, ord=2)
        if self.is_minimize():
            alpha_tilde = -(p**2) / self.penalty - predicted_diff
        else:
            alpha_tilde = p**2 / self.penalty - predicted_diff

        self.e_v = min(self.e_v, p + abs(alpha_tilde))
        alpha = self._get_alpha(cuts_list)
        if (
            alpha is not None
            and abs(alpha) > max(self.e_v, abs(10 * predicted_diff))
            and self.iter_since_update < -3
        ):
            u = 2 * self.penalty * (1 - (obj_val - center_val) / predicted_diff)
        else:
            u = self.penalty

        newu = min(u, 10 * self.penalty)
        self.iter_since_update = min(self.iter_since_update - 1, -1)
        if abs(newu - self.penalty) > BM_ABS_TOLERANCE:
            self.penalty = newu
            self.iter_since_update = -1
            self._update_center(self.center)

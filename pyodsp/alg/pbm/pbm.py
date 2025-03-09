from typing import List, Tuple
from pathlib import Path
import time

import pandas as pd
from pyomo.environ import Var, Reals, RangeSet

from pyodsp.alg.cuts import CutList

from .logger import PbmLogger
from ..bm.bm import BundleMethod
from ..params import BM_ABS_TOLERANCE, BM_REL_TOLERANCE, BM_TIME_LIMIT
from ..const import *
from pyodsp.solver.pyomo_solver import PyomoSolver
from pyodsp.solver.pyomo_utils import (
    add_quad_terms_to_objective,
    update_quad_terms_in_objective,
)

"""
TODO: As of now this is just a regularized method. Update alg to:
Kiwiel, K. C. (1990). 
Proximity control in bundle methods for convex nondifferentiable minimization. 
Mathematical programming, 46(1), 105-122.

"""


class ProximalBundleMethod(BundleMethod):
    def __init__(self, solver: PyomoSolver, max_iteration=1000, penalty=1.0) -> None:
        super().__init__(solver, max_iteration)
        self.penalty = penalty
        self.center_val = []
    
    def set_logger(self, node_id: int, depth: int) -> None:
        self.logger = PbmLogger(node_id, depth)

    def set_init_solution(self, solution: List[float]) -> None:
        self.center = solution

    def build(self, num_cuts: int, subobj_bounds: List[float] | None = None) -> None:
        self.num_cuts = num_cuts
        if subobj_bounds is not None:
            assert self.num_cuts == len(subobj_bounds)
        self._update_objective(subobj_bounds)
        self.cuts_manager.build(self.num_cuts)

        self.logger.log_initialization(
            tolerance=BM_ABS_TOLERANCE, max_iteration=self.max_iteration
        )

    def run_step(self, cuts_list: List[CutList] | None) -> Tuple[int, List[float]]:
        if cuts_list is not None:
            no_cuts, obj_val = self.add_cuts(cuts_list)
            if self.feasible:
                self.obj_val.append(obj_val)
            else:
                self.obj_val.append(None)
                
            if no_cuts or self._improved():
                self._update_center(self.current_solution)
                self.center_val.append(self.obj_val[-1])
            elif len(self.center_val) == 0:
                self.center_val.append(self.obj_val[-1])
            else:
                self.center_val.append(self.center_val[-1])
        else:
            self.obj_val.append(None)
            self.center_val.append(None)

        self._increment()

        self._solve()
        if self.status == STATUS_INFEASIBLE:
            self.logger.log_infeasible()
            return self.status, None
        self._log()

        if self._termination_check():
            self.logger.log_completion(self.iteration, self.obj_bound[-1])

        return self.status, self.current_solution
    
    def _log(self) -> None:
        if self.solver.is_minimize():
            lb = self.obj_bound[-1]
            ub = self.obj_val[-1]
        else:
            lb = self.obj_val[-1]
            ub = self.obj_bound[-1]
        numcuts = self.cuts_manager.get_num_cuts()
        elapsed = time.time() - self.start_time
        self.logger.log_master_problem(
            self.iteration, lb, self.center_val[-1], ub, self.current_solution, numcuts, elapsed
        )
    
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

        gap = abs(self.obj_bound[-1] - self.center_val[-1]) / abs(self.center_val[-1])

        if gap < BM_REL_TOLERANCE:
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
        self.solver.save(dir)

    def _improved(self) -> bool:
        if len(self.center_val) == 0 or self.center_val[-1] is None:
            return False

        if self.solver.is_minimize():
            # Minimization
            return self.obj_val[-1] <= self.center_val[-1]
        else:
            # Maximization
            return self.obj_val[-1] >= self.center_val[-1]

    def _update_objective(self, subobj_bounds: List[float] | None):
        def theta_bounds(model, i):
            if subobj_bounds is None:
                return (None, None)
            if self.solver.is_minimize():
                # Minimization
                return (subobj_bounds[i], None)
            else:
                # Maximization
                return (None, subobj_bounds[i])

        self.solver.model._theta = Var(
            RangeSet(0, self.num_cuts - 1), domain=Reals, bounds=theta_bounds
        )

        add_quad_terms_to_objective(
            self.solver,
            self.solver.model._theta,
            self.solver.vars,
            self.center,
            self.penalty,
        )

    def _update_center(self, center: List[float]) -> None:
        self.center = center
        update_quad_terms_in_objective(
            self.solver, self.solver.vars, self.center, self.penalty
        )

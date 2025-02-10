from typing import List
from pathlib import Path

import pandas as pd
from pyomo.environ import Var, Reals, RangeSet

from pyodec.alg.cuts import CutList

from .logger import PbmLogger
from ..bm.bm import BundleMethod
from ..const import BM_ABS_TOLERANCE, BM_REL_TOLERANCE
from pyodec.solver.pyomo_solver import PyomoSolver
from pyodec.solver.pyomo_utils import (
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
        self.logger = PbmLogger()

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

    def run_step(self, cuts_list: List[CutList] | None) -> List[float] | None:
        if cuts_list is not None:
            no_cuts = self._add_cuts(cuts_list)
            if no_cuts or self._improved():
                self._update_center(self.current_solution)
                self.center_val.append(self.feas_bound[-1])
            elif len(self.center_val) == 0:
                self.center_val.append(self.feas_bound[-1])
            else:
                self.center_val.append(self.center_val[-1])
        else:
            self.feas_bound.append(None)
            self.center_val.append(None)

        reached_max_iteration = self._increment()
        if reached_max_iteration:
            self.logger.log_status_max_iter()
            self.logger.log_completion(self.iteration, self.relax_bound[-1])
            return

        self._solve()

        if self.solver.is_minimize():
            lb = self.relax_bound[-1]
            ub = self.feas_bound[-1]
        else:
            lb = self.feas_bound[-1]
            ub = self.relax_bound[-1]
        self.logger.log_master_problem(
            self.iteration, lb, self.center_val[-1], ub, self.current_solution
        )

        if self._optimal():
            if len(self.relax_bound) > len(self.center_val):
                self.center_val.append(self.center_val[-1])
            if len(self.relax_bound) > len(self.feas_bound):
                self.feas_bound.append(self.feas_bound[-1])
            self.logger.log_status_optimal()
            self.logger.log_completion(self.iteration, self.relax_bound[-1])
            return

        return self.current_solution

    def save(self, dir: Path) -> None:
        path = dir / "pbm.csv"
        df = pd.DataFrame(
            {
                "obj_bound": self.relax_bound,
                "center_val": self.center_val,
                "obj_val": self.feas_bound,
            }
        )
        df.to_csv(path)
        self.solver.save(dir)

    def _improved(self) -> bool:
        if len(self.center_val) == 0 or self.center_val[-1] is None:
            return False

        if self.solver.is_minimize():
            # Minimization
            return self.feas_bound[-1] <= self.center_val[-1]
        else:
            # Maximization
            return self.feas_bound[-1] >= self.center_val[-1]

    def _optimal(self) -> bool:
        if len(self.center_val) == 0 or self.center_val[-1] is None:
            return False

        gap = abs(self.relax_bound[-1] - self.center_val[-1]) / abs(self.center_val[-1])

        return gap < BM_REL_TOLERANCE

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

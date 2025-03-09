from typing import List, Tuple
from pathlib import Path
import time

import pandas as pd

from pyomo.environ import Var, Constraint, Reals, RangeSet

from pyodsp.solver.pyomo_solver import PyomoSolver
from pyodsp.solver.pyomo_utils import add_terms_to_objective

from ..cuts import CutList, OptimalityCut, FeasibilityCut
from ..cuts_manager import CutsManager, CutInfo
from .logger import BmLogger
from ..params import BM_ABS_TOLERANCE, BM_REL_TOLERANCE, BM_PURGE_FREQ, BM_TIME_LIMIT
from ..const import *


class BundleMethod:
    def __init__(self, solver: PyomoSolver, max_iteration=1000) -> None:
        self.solver = solver
        self.cuts_manager = CutsManager()

        self.max_iteration = max_iteration
        self.iteration = 0

        self.current_solution: List[float] = []

        self.obj_bound: List[float | None] = []
        self.obj_val: List[float | None] = []

        self.status: int = STATUS_NOT_FINISHED
        self.start_time = time.time()

    def set_logger(self, node_id: int, depth: int) -> None:
        self.logger = BmLogger(node_id, depth)

    def build(self, num_cuts: int, subobj_bounds: List[float] | None) -> None:
        self.num_cuts = num_cuts
        assert subobj_bounds is not None
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
        else:
            self.obj_val.append(None)

        self._increment()
        self._solve()
        if self.status == STATUS_INFEASIBLE:
            self.logger.log_infeasible()
            return self.status, None
        self._log()
        if self._termination_check():
            self.logger.log_completion(self.iteration, self.obj_bound[-1])

        return self.status, self.current_solution

    def reset_iteration(self, i=0) -> None:
        self.iteration = i
        self.status = STATUS_NOT_FINISHED
        self.start_time = time.time()

    def _solve(self) -> None:
        self.solver.solve()
        if self.solver.is_infeasible():
            self.status = STATUS_INFEASIBLE
        self.current_solution = self.solver.get_solution()
        current_obj = self.solver.get_original_objective_value()
        if not self.solver.is_optimal():
            raise ValueError("invalid solver status")
        for idx in range(self.num_cuts):
            theta = self.solver.model._theta[idx]
            theta_val = theta.value
            current_obj += theta_val
        self.obj_bound.append(current_obj)
    
    def _log(self) -> None:
        if self.solver.is_minimize():
            lb = self.obj_bound[-1]
            ub = self.obj_val[-1]
        else:
            lb = self.obj_val[-1]
            ub = self.obj_bound[-1]
        numcuts = self.cuts_manager.get_num_cuts()
        elapsed = time.time() - self.start_time
        self.logger.log_master_problem(self.iteration, lb, ub, self.current_solution, numcuts, elapsed)

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

        gap = abs(self.obj_bound[-1] - self.obj_val[-1]) / abs(self.obj_val[-1])

        if gap < BM_REL_TOLERANCE:
            self.status = STATUS_OPTIMAL
            self.logger.log_status_optimal()
            return True
        
        return False

    def save(self, dir: Path) -> None:
        path = dir / "bm.csv"
        df = pd.DataFrame({"obj_bound": self.obj_bound, "obj_val": self.obj_val})
        df.to_csv(path)
        self.solver.save(dir)

    def add_cuts(self, cuts_list: List[CutList]) -> Tuple[bool, float]:
        found_cuts = [False for _ in range(self.num_cuts)]
        self.feasible = True
        obj_val = self.solver.get_original_objective_value()
        for idx, cuts in enumerate(cuts_list):
            for cut in cuts:
                found_cut = False
                if isinstance(cut, OptimalityCut):
                    found_cut = self._add_optimality_cut(idx, cut)
                    obj_val += cut.objective_value
                elif isinstance(cut, FeasibilityCut):
                    found_cut = self._add_feasibility_cut(idx, cut)
                    self.feasible = False
                found_cuts[idx] = found_cut or found_cuts[idx]

        optimal = not any(found_cuts)
        return optimal, obj_val

    def _increment(self) -> None:
        self.iteration += 1
        self.cuts_manager.increment()
        if self.iteration % BM_PURGE_FREQ == 0:
            self.cuts_manager.purge(self.solver.model)

    def _update_objective(self, subobj_bounds: List[float]):
        def theta_bounds(model, i):
            if self.solver.is_minimize():
                # Minimization
                return (subobj_bounds[i], None)
            else:
                # Maximization
                return (None, subobj_bounds[i])

        self.solver.model._theta = Var(
            RangeSet(0, self.num_cuts - 1), domain=Reals, bounds=theta_bounds
        )

        add_terms_to_objective(self.solver, self.solver.model._theta)

    def _add_optimality_cut(self, idx: int, cut: OptimalityCut) -> bool:

        theta = self.solver.model._theta[idx]
        theta_val = theta.value
        cut_num = self.cuts_manager.get_num_optimality(idx)
        vars = self.solver.vars

        if self.solver.is_minimize():
            # Minimization
            if (
                theta_val is not None
                and theta_val >= cut.objective_value - BM_ABS_TOLERANCE
            ):
                # No need to add the cut
                return False

            constraint = Constraint(
                expr=sum(coeff * vars[j] for j, coeff in cut.coeffs.items()) + theta
                >= cut.rhs
            )
        else:
            # Maximization
            if (
                theta_val is not None
                and theta_val <= cut.objective_value + BM_ABS_TOLERANCE
            ):
                # No need to add the cut
                return False

            constraint = Constraint(
                expr=sum(coeff * vars[j] for j, coeff in cut.coeffs.items()) + theta
                <= cut.rhs
            )

        self.solver.model.add_component(f"_optimality_cut_{idx}_{cut_num}", constraint)

        self.cuts_manager.append_cut(
            CutInfo(constraint, cut, idx, self.iteration, self.current_solution, 0)
        )

        return True

    def _add_feasibility_cut(self, idx: int, cut: FeasibilityCut) -> bool:

        cut_num = self.cuts_manager.get_num_feasibility(idx)
        vars = self.solver.vars

        if self.solver.is_minimize():
            # Minimization
            constraint = Constraint(
                expr=sum(coeff * vars[j] for j, coeff in cut.coeffs.items()) >= cut.rhs
            )
        else:
            # Maximization
            constraint = Constraint(
                expr=sum(coeff * vars[j] for j, coeff in cut.coeffs.items()) <= cut.rhs
            )
        self.solver.model.add_component(f"_feasibility_cut_{idx}_{cut_num}", constraint)

        self.cuts_manager.append_cut(
            CutInfo(constraint, cut, idx, self.iteration, self.current_solution, 0)
        )

        return True

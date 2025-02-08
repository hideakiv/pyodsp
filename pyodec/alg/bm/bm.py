from typing import List

from pyomo.environ import Var, Constraint, Reals, RangeSet

from pyodec.solver.pyomo_solver import PyomoSolver
from pyodec.solver.pyomo_utils import add_terms_to_objective

from .cuts import CutList, OptimalityCut, FeasibilityCut
from .cuts_manager import CutsManager, CutInfo
from .logger import BmLogger
from ..const import BM_ABS_TOLERANCE


class BundleMethod:
    def __init__(self, solver: PyomoSolver, max_iteration=1000) -> None:
        self.solver = solver
        self.cuts_manager = CutsManager()

        self.max_iteration = max_iteration
        self.iteration = 1
        self.logger = BmLogger()

        self.current_solution: List[float] = []

        self.relax_bound: List[float | None] = []
        self.feas_bound: List[float | None] = []

    def build(self, subobj_bounds: List[float]) -> None:
        self.num_cuts = len(subobj_bounds)
        self._update_objective(subobj_bounds)
        self.cuts_manager.build(self.num_cuts)

        self.logger.log_initialization(
            tolerance=BM_ABS_TOLERANCE, max_iteration=self.max_iteration
        )

    def reset_iteration(self, i=1) -> None:
        self.iteration = i

    def solve(self) -> None:
        self.solver.solve()
        self.current_solution = self.solver.get_solution()
        self.relax_bound.append(self.solver.get_objective_value())
        if self.solver.is_minimize():
            lb = self.relax_bound[-1]
            ub = self.feas_bound[-1]
        else:
            lb = self.feas_bound[-1]
            ub = self.relax_bound[-1]
        self.logger.log_master_problem(self.iteration, lb, ub, self.current_solution)

    def get_solution(self) -> List[float]:
        return self.current_solution

    def add_cuts(self, cuts_list: List[CutList]) -> bool:
        found_cuts = [False for _ in range(self.num_cuts)]
        feasible = True
        obj_val = 0.0
        for idx, cuts in enumerate(cuts_list):
            for cut in cuts:
                found_cut = False
                if isinstance(cut, OptimalityCut):
                    found_cut = self._add_optimality_cut(idx, cut)
                    obj_val += cut.objective_value
                elif isinstance(cut, FeasibilityCut):
                    found_cut = self._add_feasibility_cut(idx, cut)
                    feasible = False
                found_cuts[idx] = found_cut or found_cuts[idx]

        if feasible:
            self.feas_bound.append(obj_val)
        else:
            self.feas_bound.append(None)

        optimal = not any(found_cuts)
        if optimal:
            self.logger.log_status_optimal()
            self.logger.log_completion(self.iteration, self.relax_bound[-1])
            return True

        reached_max_iteration = self.increment()
        if reached_max_iteration:
            self.logger.log_status_max_iter()
            self.logger.log_completion(self.iteration, self.relax_bound[-1])
            return True

        return False

    def increment(self) -> bool:
        self.iteration += 1
        self.cuts_manager.increment()

        # reached max iteration
        return self.iteration > self.max_iteration

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
                expr=sum(cut.coeffs[j] * vars[j] for j in range(len(vars))) + theta
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
                expr=sum(cut.coeffs[j] * vars[j] for j in range(len(vars))) + theta
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
                expr=sum(cut.coeffs[j] * vars[j] for j in range(len(vars))) >= cut.rhs
            )
        else:
            # Maximization
            constraint = Constraint(
                expr=sum(cut.coeffs[j] * vars[j] for j in range(len(vars))) <= cut.rhs
            )
        self.solver.model.add_component(f"_feasibility_cut_{idx}_{cut_num}", constraint)

        self.cuts_manager.append_cut(
            CutInfo(constraint, cut, idx, self.iteration, self.current_solution, 0)
        )

        return True

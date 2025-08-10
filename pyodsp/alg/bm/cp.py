from pathlib import Path

from pyomo.environ import ScalarVar, Constraint

from pyodsp.solver.pyomo_solver import PyomoSolver
from ..cuts_manager import CutsManager, CutInfo
from ..cuts import CutList, OptimalityCut, FeasibilityCut

from ..params import BM_ABS_TOLERANCE


class CuttingPlaneMethod:
    def __init__(self, solver: PyomoSolver) -> None:
        self.solver = solver
        self.cuts_manager = CutsManager()
        self.current_solution: list[float] = []

    def is_minimize(self) -> bool:
        return self.solver.is_minimize()

    def get_cuts(self) -> list[list[CutInfo]]:
        return self.cuts_manager.get_cuts()

    def get_num_cuts(self) -> int:
        return self.cuts_manager.get_num_cuts()

    def get_vars(self) -> list[ScalarVar]:
        return self.solver.get_vars()

    def get_original_objective_value(self) -> float:
        return self.solver.get_original_objective_value()

    def get_solver(self) -> PyomoSolver:
        return self.solver

    def build(self, num_cuts: int) -> None:
        self.num_cuts = num_cuts
        self.cuts_manager.build(num_cuts)

    def is_infeasible(self):
        return self.solver.is_infeasible()

    def solve(self) -> None:
        self.solver.solve()
        self.current_solution = self.solver.get_solution()

    def get_current_solution(self) -> list[float]:
        return self.current_solution

    def get_theta_value(self, idx: int) -> float:
        theta = self.solver.model._theta[idx]
        return theta.value

    def get_relaxed_objective(self) -> float:
        if not self.solver.is_optimal():
            raise ValueError("invalid solver status")
        current_obj = self.get_original_objective_value()
        for idx in range(self.num_cuts):
            current_obj += self.get_theta_value(idx)
        return current_obj

    def add_cuts(self, cuts_list: list[CutList]) -> tuple[bool, bool, float]:
        found_cuts = [False for _ in range(self.num_cuts)]
        feasible = True
        obj_val = self.solver.get_original_objective_value()
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

        optimal = not any(found_cuts)
        return optimal, feasible, obj_val

    def _add_optimality_cut(self, idx: int, cut: OptimalityCut) -> bool:
        theta = self.solver.model._theta[idx]
        theta_val = theta.value
        cut_num = self.cuts_manager.get_num_optimality(idx)
        vars = self.get_vars()

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
            CutInfo(constraint, cut, idx, self.current_solution, 0)
        )

        return True

    def _add_feasibility_cut(self, idx: int, cut: FeasibilityCut) -> bool:
        cut_num = self.cuts_manager.get_num_feasibility(idx)
        vars = self.get_vars()

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
            CutInfo(constraint, cut, idx, self.current_solution, 0)
        )

        return True

    def increment_cuts(self) -> None:
        self.cuts_manager.increment()

    def purge_cuts(self) -> None:
        self.cuts_manager.purge(self.solver.model)

    def save(self, dir: Path) -> None:
        self.solver.save(dir)

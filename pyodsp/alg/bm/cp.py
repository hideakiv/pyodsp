import json
from pathlib import Path

import pandas as pd
from pyomo.environ import ScalarVar, Constraint, Objective, minimize

from pyodsp.solver.pyomo_solver import PyomoSolver
from .cuts_manager import CutsManager, NonPurgingCutsManager, CutInfo
from .cuts import CutList, OptimalityCut, FeasibilityCut

from ..params import BM_ABS_TOLERANCE

CUTS_FILE = "cuts.csv"


class CuttingPlaneMethod:
    """Cutting-plane master problem, always solved as a minimization.

    A maximize problem is handled by negating (via `self._sign`) the true
    objective's contribution, all incoming cut coefficients/rhs/objective
    values, and reported theta/objective values back on the way out — the
    cut/theta bookkeeping itself never branches on optimization direction.
    """

    def __init__(
        self, solver: PyomoSolver, force: bool = False, purgeable: bool = True
    ) -> None:
        self.solver = solver
        self.cuts_manager = CutsManager() if purgeable else NonPurgingCutsManager()
        self.current_solution: list[float] = []
        self.force = force
        self._sign = 1.0 if solver.is_minimize() else -1.0

    def is_minimize(self) -> bool:
        return self.solver.is_minimize()

    def get_sign(self) -> float:
        return self._sign

    def build_theta_objective(self, theta_vars) -> None:
        """Wire theta into the master's working objective (`_mod_obj`),
        which is always solved as minimize regardless of the true sense.
        """
        solver = self.solver
        solver.original_objective.deactivate()
        if solver.model.component("_mod_obj") is not None:
            solver.model.del_component("_mod_obj")
        modified_expr = self._sign * solver.original_objective.expr + sum(
            theta_vars[i] for i in range(len(theta_vars))
        )
        solver.model._mod_obj = Objective(expr=modified_expr, sense=minimize)

    def get_cuts(self) -> list[list[CutInfo]]:
        return self.cuts_manager.get_cuts()

    def get_num_cuts(self) -> int:
        return self.cuts_manager.get_num_cuts()

    def get_vars(self) -> list[ScalarVar]:
        return self.solver.get_vars()

    def get_original_objective_value(self) -> float | None:
        return self.solver.get_original_objective_value()

    def get_objective_value(self) -> float:
        return self._sign * self.solver.get_objective_value()

    def get_parent_objective_value(self) -> float:
        return self.solver.get_parent_objective_value()

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

    def get_theta_value(self, idx: int) -> float | None:
        theta = self.solver.model._theta[idx]
        if theta.value is None:
            return None
        return self._sign * theta.value

    def get_relaxed_objective(self) -> float | None:
        if not self.solver.is_optimal():
            raise ValueError("invalid solver status")
        current_obj = self.get_original_objective_value()
        if current_obj is None:
            return None

        for idx in range(self.num_cuts):
            current_obj += self.get_theta_value(idx)
        return current_obj

    def add_cuts(self, cuts_list: list[CutList]) -> tuple[bool, bool, float | None]:
        found_cuts = [False for _ in range(self.num_cuts)]
        feasible = True
        obj_val = self.get_original_objective_value()
        for idx, cuts in enumerate(cuts_list):
            for cut in cuts:
                found_cut = False
                if isinstance(cut, OptimalityCut):
                    found_cut = self._add_optimality_cut(idx, cut)
                    if obj_val is not None:
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
        sign = self._sign

        if (
            not self.force
            and theta_val is not None
            and theta_val >= sign * cut.objective_value - BM_ABS_TOLERANCE
        ):
            # No need to add the cut
            return False

        constraint = Constraint(
            expr=sum(sign * coeff * vars[j] for j, coeff in cut.coeffs.items()) + theta
            >= sign * cut.rhs
        )

        self.solver.model.add_component(f"_optimality_cut_{idx}_{cut_num}", constraint)

        self.cuts_manager.append_cut(
            CutInfo(constraint, cut, idx, self.current_solution, 0)
        )

        return True

    def _add_feasibility_cut(self, idx: int, cut: FeasibilityCut) -> bool:
        cut_num = self.cuts_manager.get_num_feasibility(idx)
        vars = self.get_vars()
        sign = self._sign

        constraint = Constraint(
            expr=sum(sign * coeff * vars[j] for j, coeff in cut.coeffs.items())
            >= sign * cut.rhs
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

    def eliminate_cuts(self, names: list[str]) -> None:
        self.cuts_manager.eliminate_cuts(self.solver.model, names)

    def save(self, dir: Path) -> None:
        self.solver.save(dir)
        self.save_cuts(dir)

    def save_cuts(self, dir: Path) -> None:
        """Write the active cuts to cuts.csv.

        These are the durable part of what a master has learned: the cuts
        describe its value function over the coupling variables, and hold
        for any state, unlike sol.csv or a dumped model — both of which
        are conditioned on whatever the coupling variables happened to be
        fixed to at the last solve. Coefficients are keyed by coupling
        variable name so the file stands on its own; load_cuts maps them
        back to positions.
        """
        var_names = [var.name for var in self.get_vars()]
        rows = []
        for group in self.get_cuts():
            for cut_info in group:
                cut = cut_info.cut
                is_optimality = isinstance(cut, OptimalityCut)
                rows.append(
                    {
                        "idx": cut_info.idx,
                        "type": "optimality" if is_optimality else "feasibility",
                        "rhs": cut.rhs,
                        "objective_value": (
                            cut.objective_value if is_optimality else None
                        ),
                        "coeffs": json.dumps(
                            {var_names[j]: coeff for j, coeff in cut.coeffs.items()}
                        ),
                    }
                )
        df = pd.DataFrame(
            rows, columns=["idx", "type", "rhs", "objective_value", "coeffs"]
        )
        df.to_csv(Path(dir) / CUTS_FILE, index=False)

    def load_cuts(self, dir: Path) -> list[CutList]:
        """Read cuts.csv back into one CutList per theta index.

        Coefficients are matched to this master's coupling variables by
        name, so a saved value function can be reinstated onto a rebuilt
        model (see BundleMethod.restore_cuts) as long as it couples on the
        same variables — their order is allowed to differ.
        """
        df = pd.read_csv(Path(dir) / CUTS_FILE)
        positions = {var.name: j for j, var in enumerate(self.get_vars())}
        cuts_list: list[CutList] = [CutList() for _ in range(self.num_cuts)]
        for row in df.itertuples(index=False):
            coeffs = {}
            for name, coeff in json.loads(row.coeffs).items():
                if name not in positions:
                    raise ValueError(
                        f"Saved cut references coupling variable '{name}', which "
                        "this master does not have"
                    )
                coeffs[positions[name]] = float(coeff)
            idx = int(row.idx)
            if idx >= self.num_cuts:
                raise ValueError(
                    f"Saved cut belongs to group {idx}, but this master only has "
                    f"{self.num_cuts} group(s) — it was built with different groups"
                )
            if row.type == "optimality":
                cut = OptimalityCut(
                    coeffs=coeffs,
                    rhs=float(row.rhs),
                    info={},
                    objective_value=float(row.objective_value),
                )
            else:
                cut = FeasibilityCut(coeffs=coeffs, rhs=float(row.rhs), info={})
            cuts_list[idx].append(cut)
        return cuts_list

import json
from pathlib import Path

import pandas as pd
from pyomo.environ import (
    ScalarVar,
    Constraint,
    Objective,
    RangeSet,
    Reals,
    Var,
    minimize,
)

from pyodsp.solver.pyomo_solver import PyomoSolver
from .cuts_manager import CutsManager, NonPurgingCutsManager, CutInfo
from .cuts import CutList, OptimalityCut, FeasibilityCut

from ..params import BM_ABS_TOLERANCE
from ..risk import Expectation, RiskMeasure, value_of_sample

CUTS_FILE = "cuts.csv"


class CuttingPlaneMethod:
    """Cutting-plane master problem, always solved as a minimization.

    A maximize problem is handled by negating (via `self._sign`) the true
    objective's contribution, all incoming cut coefficients/rhs/objective
    values, and reported theta/objective values back on the way out — the
    cut/theta bookkeeping itself never branches on optimization direction.
    """

    def __init__(
        self,
        solver: PyomoSolver,
        force: bool = False,
        purgeable: bool = True,
        risk: RiskMeasure | None = None,
    ) -> None:
        self.solver = solver
        self.risk: RiskMeasure = risk or Expectation()
        # Probabilities per theta, needed to state the tail. Supplied at
        # build time, since a master does not know its children until then.
        self.group_probabilities: list[float] | None = None
        self.cuts_manager = CutsManager() if purgeable else NonPurgingCutsManager()
        self.current_solution: list[float] = []
        self.force = force
        self._sign = 1.0 if solver.is_minimize() else -1.0
        self._user_sense_multiplier: float | None = None

    def is_minimize(self) -> bool:
        return self.solver.is_minimize()

    def get_sign(self) -> float:
        return self._sign

    def build_theta_objective(self, theta_vars, group_probabilities=None) -> None:
        """Wire theta into the master's working objective (`_mod_obj`),
        which is always solved as minimize regardless of the true sense.

        Under a risk measure the future is not priced by its mean, so the
        sum of thetas is replaced — see _risk_expression.
        """
        solver = self.solver
        self.group_probabilities = (
            list(group_probabilities) if group_probabilities is not None else None
        )
        solver.original_objective.deactivate()
        if solver.model.component("_mod_obj") is not None:
            solver.model.del_component("_mod_obj")

        future = self._risk_expression(theta_vars)
        modified_expr = self._sign * solver.original_objective.expr + future
        solver.model._mod_obj = Objective(expr=modified_expr, sense=minimize)

    def _risk_expression(self, theta_vars):
        """What the master pays for the future.

        Risk-neutrally that is just the sum of the thetas: each already
        carries its scenario's probability (CutAggregator scales every cut
        by it), so they add up to an expectation.

        For CVaR it is the Rockafellar-Uryasev program

            (1-w) * sum_s theta_s + w * ( eta + 1/(1-a) * sum_s zeta_s )
            zeta_s >= theta_s - p_s * eta,   zeta_s >= 0

        stated in probability-weighted terms throughout, so nothing is
        ever divided by a scenario probability. `eta` is a genuine
        first-stage decision — the value at risk — solved for alongside
        the rest.
        """
        total = sum(theta_vars[i] for i in range(len(theta_vars)))
        risk = self.risk
        if isinstance(risk, Expectation) or risk.is_risk_neutral:
            return total

        probabilities = self.group_probabilities
        if probabilities is None or len(probabilities) != len(theta_vars):
            raise ValueError(
                "A risk measure needs one probability per theta, and this "
                "master was built without them. Every scenario must be its "
                "own cut group: a risk measure prices the spread across "
                "scenarios, which an aggregated cut has already averaged "
                "away."
            )

        model = self.solver.model
        indices = RangeSet(0, len(theta_vars) - 1)
        for name in ("_eta", "_zeta", "_zeta_constraint"):
            if model.component(name) is not None:
                model.del_component(name)

        model._eta = Var(domain=Reals)
        model._zeta = Var(indices, domain=Reals, bounds=(0.0, None))

        def zeta_rule(_, i):
            return model._zeta[i] >= theta_vars[i] - probabilities[i] * model._eta

        model._zeta_constraint = Constraint(indices, rule=zeta_rule)

        tail = model._eta + (1.0 / risk.tail) * sum(
            model._zeta[i] for i in range(len(theta_vars))
        )
        return (1.0 - risk.weight) * total + risk.weight * tail

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

    def set_sense_multiplier(self, multiplier: float) -> None:
        """Report in this sense instead of the master model's own.

        For a master synthesized rather than supplied — dual
        decomposition's Lagrangian master — the model here says nothing
        about the sense the user wrote, so the caller supplies it.
        """
        self._user_sense_multiplier = multiplier

    def get_sense_multiplier(self) -> float:
        """-1.0 if the problem this master serves was written as maximize.

        Distinct from `_sign`, which is about *this* master's own Pyomo
        sense — dual decomposition's Lagrangian master is deliberately a
        maximize model and has `_sign == -1` while serving a problem whose
        multiplier may be either.
        """
        if self._user_sense_multiplier is not None:
            return self._user_sense_multiplier
        return self.solver.sense_multiplier

    def to_user_units(self, value: float | None) -> float | None:
        """One recorded value converted back to the caller's sense."""
        return None if value is None else self.get_sense_multiplier() * value

    def in_user_units(self, values: list[float | None]) -> list[float | None]:
        """A recorded trajectory converted back to the caller's sense.

        The iteration bookkeeping runs entirely in the converted (minimize)
        units, so this is applied once, where values leave the algorithm.
        """
        if self.get_sense_multiplier() > 0.0:
            return list(values)
        return [None if v is None else -v for v in values]

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
        # Each group's realized cost at this trial point, kept per group so
        # a risk measure can price the spread. Summing them here instead
        # would be an expectation, which under CVaR is not the quantity the
        # master is minimizing — the convergence test would then compare a
        # risk-adjusted bound against a risk-neutral incumbent.
        realized: list[float | None] = [None] * self.num_cuts
        for idx, cuts in enumerate(cuts_list):
            for cut in cuts:
                found_cut = False
                if isinstance(cut, OptimalityCut):
                    found_cut = self._add_optimality_cut(idx, cut)
                    # Accumulated, not assigned: a group normally holds one
                    # aggregated cut, but replace_cuts and BDSC's column
                    # reuse both hand over every cut a group has at once,
                    # and the total is what the previous behaviour was.
                    realized[idx] = (realized[idx] or 0.0) + cut.objective_value
                elif isinstance(cut, FeasibilityCut):
                    found_cut = self._add_feasibility_cut(idx, cut)
                    feasible = False
                found_cuts[idx] = found_cut or found_cuts[idx]

        if obj_val is not None:
            future = self._realized_future(realized)
            obj_val = None if future is None else obj_val + future

        optimal = not any(found_cuts)
        return optimal, feasible, obj_val

    def _realized_future(self, realized: list[float | None]) -> float | None:
        """The future's cost at this trial point, valued as the master does."""
        present = [value for value in realized if value is not None]
        risk = self.risk
        if isinstance(risk, Expectation) or risk.is_risk_neutral:
            return sum(present)
        if len(present) != self.num_cuts or self.group_probabilities is None:
            # A feasibility cut means some scenario had no cost to report,
            # so there is no distribution to take the tail of yet.
            return None
        probabilities = self.group_probabilities
        # The cuts carry probability-weighted costs; the tail is over the
        # costs themselves.
        costs = [value / p for value, p in zip(present, probabilities)]
        return value_of_sample(risk, costs, probabilities)

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
        if not self.solver.has_solution():
            # Ageing judges a cut by its slack at the current point, and
            # there is no point yet. Cuts can be in the master before its
            # first solve — seeded from a previous master (see
            # BdScAlgLeafPyomo._create_master) or restored from disk (see
            # BundleMethod.restore_cuts) — and ageing them against an
            # unsolved model would both fail to evaluate and, if it could,
            # retire cuts that were never observed slack.
            return
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

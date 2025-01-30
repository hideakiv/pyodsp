from typing import List, Dict

from pyomo.environ import ConcreteModel, Var, Constraint, Objective, Reals, RangeSet
from pyomo.core.base.var import VarData

from .cuts import Cut, OptimalityCut, FeasibilityCut, WrappedCut
from ..const import BM_ABS_TOLERANCE


class BundleManager:
    def __init__(self, model: ConcreteModel) -> None:
        self.model = model
        self.active_cuts: List[WrappedCut] = []

        self.optimality_cuts: Dict[int, int] = {}
        self.feasibility_cuts: Dict[int, int] = {}

    def build(
        self,
        subobj_bounds: List[float],
        original_objective: Objective,
        is_minimize: bool,
    ):
        self.num_cuts = len(subobj_bounds)
        self.original_objective = original_objective
        self.is_minimize = is_minimize
        self._update_objective(subobj_bounds)
        for i in range(self.num_cuts):
            self.optimality_cuts[i] = 0
            self.feasibility_cuts[i] = 0

    def _update_objective(self, subobj_bounds: List[float]):
        def theta_bounds(model, i):
            if self.is_minimize:
                # Minimization
                return (subobj_bounds[i], None)
            else:
                # Maximization
                return (None, subobj_bounds[i])

        self.model._theta = Var(
            RangeSet(0, self.num_cuts - 1), domain=Reals, bounds=theta_bounds
        )

        self.original_objective.deactivate()

        modified_expr = self.original_objective.expr + sum(
            self.model._theta[i] for i in range(self.num_cuts)
        )

        self.model._mod_obj = Objective(
            expr=modified_expr, sense=self.original_objective.sense
        )

    def _append_cut(self, cut: Cut, iteration: int, trial_point: List[float]):
        wrapped_cut = WrappedCut(cut, iteration, trial_point, 0)
        self.active_cuts.append(wrapped_cut)

    def add_optimality_cut(
        self,
        iteration: int,
        i: int,
        cut: OptimalityCut,
        vars: List[VarData],
        trial_point: List[float],
    ) -> bool:

        theta_val = self.model._theta[i].value
        cut_num = self.optimality_cuts[i]

        if self.is_minimize:
            # Minimization
            if theta_val >= cut.objective_value - BM_ABS_TOLERANCE:
                # No need to add the cut
                return False
            self.model.add_component(
                f"_optimality_cut_{i}_{cut_num}",
                Constraint(
                    expr=sum(cut.coeffs[i] * vars[i] for i in range(len(vars)))
                    + self.model._theta[i]
                    >= cut.rhs
                ),
            )
        else:
            # Maximization
            if theta_val <= cut.objective_value + BM_ABS_TOLERANCE:
                # No need to add the cut
                return False
            self.model.add_component(
                f"_optimality_cut_{i}_{cut_num}",
                Constraint(
                    expr=sum(cut.coeffs[i] * vars[i] for i in range(len(vars)))
                    + self.model._theta[i]
                    <= cut.rhs
                ),
            )

        self.optimality_cuts[i] += 1
        self._append_cut(cut, iteration, trial_point)

        return True

    def add_feasibility_cut(
        self,
        iteration: int,
        i: int,
        cut: FeasibilityCut,
        vars: List[VarData],
        trial_point: List[float],
    ) -> bool:

        cut_num = self.feasibility_cuts[i]

        if self.is_minimize:
            # Minimization
            self.model.add_component(
                f"_feasibility_cut_{i}_{cut_num}",
                Constraint(
                    expr=sum(cut.coeffs[i] * vars[i] for i in range(len(vars)))
                    >= cut.rhs
                ),
            )
        else:
            # Maximization
            self.model.add_component(
                f"_feasibility_cut_{i}_{cut_num}",
                Constraint(
                    expr=sum(cut.coeffs[i] * vars[i] for i in range(len(vars)))
                    <= cut.rhs
                ),
            )

        self.feasibility_cuts[i] += 1
        self._append_cut(cut, iteration, trial_point)

        return True

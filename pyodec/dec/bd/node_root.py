from typing import List, Dict

from pyomo.core.base.var import VarData

from .node import BdNode
from .cuts import Cut, OptimalityCut, FeasibilityCut
from .solver_root import BdSolverRoot


class BdRootNode(BdNode):
    TOLERANCE = 1e-6

    def __init__(
        self,
        idx: int,
        solver: BdSolverRoot,
        vars_dn: List[VarData],
    ) -> None:
        super().__init__(idx, parent=None)
        self.solver = solver
        self.coupling_vars_dn: List[VarData] = vars_dn

        self.children_bounds: Dict[int, float] = {}

        self.built = False

    def set_bound(self, idx, bound) -> None:
        self.children_bounds[idx] = bound

    def remove_child(self, idx):
        self.children_bounds.pop(idx)
        return super().remove_child(idx)

    def remove_children(self):
        self.children_bounds = {}
        return super().remove_children()

    def build(self, groups: List[List[int]] | None = None):
        if groups is None:
            self.groups = [[child] for child in self.children]
        else:
            self.groups = groups
        self.num_cuts = len(self.groups)

        subobj_bounds = []
        for group in self.groups:
            bound = 0.0
            for member in group:
                bound += (
                    self.children_multipliers[member] * self.children_bounds[member]
                )
            subobj_bounds.append(bound)

        self.solver.build(subobj_bounds)
        self.built = True

    def solve(self) -> None:
        self.solver.solve()

    def get_coupling_solution(self) -> List[float]:
        return self.solver.get_solution(self.coupling_vars_dn)

    def add_cuts(self, cuts: Dict[int, Cut]) -> bool:
        found_cuts = [False for _ in range(self.num_cuts)]
        for i, group in enumerate(self.groups):
            group_cut = []
            group_multipliers = []
            for child in group:
                group_cut.append(cuts[child])
                group_multipliers.append(self.children_multipliers[child])
            aggregate_cut = self._aggregate_cuts(group_multipliers, group_cut)

            for cut in aggregate_cut:
                if isinstance(cut, OptimalityCut):
                    found_cut = self.solver.add_optimality_cut(
                        i, cut, self.coupling_vars_dn
                    )
                else:
                    found_cut = self.solver.add_feasibility_cut(
                        i, cut, self.coupling_vars_dn
                    )

                found_cuts[i] = found_cut or found_cuts[i]
        optimal = not any(found_cuts)
        return optimal

    def _aggregate_cuts(self, multipliers: List[float], cuts: List[Cut]) -> List[Cut]:
        new_coef = [0 for _ in range(len(self.coupling_vars_dn))]
        new_constant = 0
        new_objective = 0
        feasibility_cuts = []
        for multiplier, cut in zip(multipliers, cuts):
            if isinstance(cut, OptimalityCut):
                if len(feasibility_cuts) == 0:
                    new_coef = [
                        new_coef[i] + multiplier * cut.coefficients[i]
                        for i in range(len(self.coupling_vars_dn))
                    ]
                    new_constant += multiplier * cut.constant
                    new_objective += multiplier * cut.objective_value
            elif isinstance(cut, FeasibilityCut):
                feasibility_cuts.append(cut)
        if len(feasibility_cuts) > 0:
            return feasibility_cuts
        return [
            OptimalityCut(
                coefficients=new_coef,
                constant=new_constant,
                objective_value=new_objective,
            )
        ]

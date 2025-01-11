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
        sub_solver: BdSolverRoot,
        vars_dn: List[VarData],
        multiplier: float = 1.0,
    ) -> None:
        super().__init__(idx, sub_solver, parent=None, multiplier=multiplier)
        self.coupling_vars_dn: List[VarData] = vars_dn

        self.built = False

    def build(self, subobj_bounds: List[float]):
        self.built = True
        self._set_num_cuts(len(subobj_bounds))

        self.assign = self._assign_children()
        self.solver.build(subobj_bounds)

    def _assign_children(self) -> List[List[int]]:
        assign = [[] for _ in range(self.num_cuts)]
        for i, child in enumerate(self.children):
            assign_i = i % self.num_cuts
            assign[assign_i].append(child)
        return assign

    def _set_num_cuts(self, num_cuts: int):
        if num_cuts < 0:
            raise ValueError("Number of cuts must be non-negative")
        if num_cuts > len(self.children):
            raise ValueError(
                "Number of cuts must be less than or equal to the number of children"
            )
        if num_cuts != 1:
            raise ValueError("Only one cut is supported at the moment")
        self.num_cuts = num_cuts

    def get_coupling_solution(self) -> List[float]:
        return self.solver.get_solution(self.coupling_vars_dn)

    def add_cuts(self, multipliers: Dict[int, float], cuts: Dict[int, Cut]) -> bool:
        found_cuts = [False for _ in range(self.num_cuts)]
        for i, group in enumerate(self.assign):
            group_cut = []
            group_multipliers = []
            for child in group:
                group_cut.append(cuts[child])
                group_multipliers.append(multipliers[child])
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

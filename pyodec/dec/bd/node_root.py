from typing import List, Dict

from pyomo.core.base.var import VarData
from pyomo.environ import Var, Constraint, Reals, RangeSet

from .node import BdNode
from .cuts import Cut, OptimalityCut, FeasibilityCut
from pyodec.core.subsolver.subsolver import SubSolver


class BdRootNode(BdNode):
    TOLERANCE = 1e-6

    def __init__(
        self,
        idx: int,
        sub_solver: SubSolver,
        vars_dn: List[VarData],
        multiplier: float = 1.0,
    ) -> None:
        super().__init__(idx, sub_solver, parent=None, multiplier=multiplier)
        self.coupling_vars_dn: List[VarData] = vars_dn
        self.cut_num = 0

        self.built = False

    def build(self, num_cuts: int | None = None):
        self.built = True
        if num_cuts is None:
            self.num_cuts = len(self.children)
        else:
            self._set_num_cuts(num_cuts)

        self.assign = self._assign_children()
        self._add_sub_objective_vars()

    def _assign_children(self) -> List[List[int]]:
        assign = [[] for _ in range(self.num_cuts)]
        for i, child in enumerate(self.children):
            assign_i = i % self.num_cuts
            assign[assign_i].append(child)
        return assign

    def _add_sub_objective_vars(self):
        self.solver.model._cut_set = RangeSet(0, self.num_cuts - 1)
        self.solver.model._theta = Var(self.solver.model._cut_set, domain=Reals)
        self.active_theta = [False for _ in range(self.num_cuts)]

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

    def add_cuts(self, cuts: Dict[int, Cut]) -> bool:
        found_cuts = [False for _ in range(self.num_cuts)]
        for i, group in enumerate(self.assign):
            if len(group) > 1:
                group_cut = []
                for child in group:
                    group_cut.append(cuts[child])
                aggregate_cut = self._aggregate_cuts(group_cut)
            else:
                aggregate_cut = [cuts[group[0]]]

            for cut in aggregate_cut:
                if isinstance(cut, OptimalityCut):
                    if not self.active_theta[i]:
                        self.solver.get_objective().expr += self.solver.model._theta[i]
                        self.active_theta[i] = True
                    else:
                        theta_val = self.solver.model._theta[i].value
                        if theta_val >= cut.objective_value - self.TOLERANCE:
                            continue
                    self.add_optimality_cut(
                        self.solver.model._theta[i], cut, self.coupling_vars_dn
                    )
                else:
                    raise NotImplementedError("only implemented feasible case")
                found_cuts[i] = True
        optimal = not any(found_cuts)
        return optimal

    def _aggregate_cuts(self, cuts: List[Cut]) -> List[Cut]:
        new_coef = [0 for _ in range(len(self.coupling_vars_dn))]
        new_constant = 0
        new_objective = 0
        feasibility_cuts = []
        for cut in cuts:
            if isinstance(cut, OptimalityCut):
                if len(feasibility_cuts) == 0:
                    new_coef = [
                        new_coef[i] + cut.coefficients[i]
                        for i in range(len(self.coupling_vars_dn))
                    ]
                    new_constant += cut.constant
                    new_objective += cut.objective_value
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

    def add_optimality_cut(
        self, theta, cut: OptimalityCut, vars: List[VarData]
    ) -> None:

        self.cut_num += 1
        self.solver.model.add_component(
            f"_cut_{self.cut_num}",
            Constraint(
                expr=sum(cut.coefficients[i] * vars[i] for i in range(len(vars)))
                + theta
                >= cut.constant
            ),
        )

from typing import List, Dict

from pyomo.environ import ConcreteModel, Var, RangeSet, Objective, minimize, maximize
from pyomo.core.base.var import VarData

from pyodec.alg.bm.cuts import Cut, OptimalityCut, FeasibilityCut, CutList
from pyodec.dec.utils import get_nonzero_coefficients_group
from pyodec.solver.pyomo_solver import PyomoSolver

from .node import DdNode
from .solver_root import DdAlgRoot


class DdRootNode(DdNode):

    def __init__(
        self,
        idx: int,
        coupling_model: ConcreteModel,
        is_minimize: bool,
        solver_name: str,
        vars_dn: Dict[int, List[VarData]],
        max_iteration=1000,
        **kwargs
    ) -> None:
        super().__init__(idx, parent=None)
        self.coupling_vars_dn: Dict[int, List[VarData]] = vars_dn
        self.is_minimize = is_minimize
        self.alg = self._create_master(
            coupling_model, solver_name, max_iteration, **kwargs
        )

        self.groups = None
        self.built = False

    def _create_master(
        self,
        coupling_model: ConcreteModel,
        solver_name: str,
        max_iteration: int,
        **kwargs
    ) -> DdAlgRoot:
        master = ConcreteModel()
        self.lagrangian_data = get_nonzero_coefficients_group(
            coupling_model, self.coupling_vars_dn
        )
        self.num_constrs = len(self.lagrangian_data.constraints)

        def _bounds_rule(m, i):
            if self.lagrangian_data.sense[i] < 0:
                return (None, 0)
            elif self.lagrangian_data.sense[i] > 0:
                return (0, None)
            else:
                return (None, None)

        master.lagrangian_dual = Var(
            RangeSet(0, self.num_constrs - 1), bounds=_bounds_rule
        )
        lagrangian_duals: List[VarData] = [
            master.lagrangian_dual[i] for i in range(self.num_constrs)
        ]
        if self.is_minimize:
            master.objective = Objective(
                expr=sum(
                    -self.lagrangian_data.rhs[i] * master.lagrangian_dual[i]
                    for i in range(self.num_constrs)
                ),
                sense=maximize,
            )
        else:
            master.objective = Objective(
                expr=sum(
                    self.lagrangian_data.rhs[i] * master.lagrangian_dual[i]
                    for i in range(self.num_constrs)
                ),
                sense=minimize,
            )
        solver = PyomoSolver(master, solver_name, lagrangian_duals, **kwargs)
        return DdAlgRoot(solver, max_iteration)

    def set_groups(self, groups: List[List[int]]):
        self.groups = groups

    def build(self) -> None:
        if self.built:
            return
        if self.groups is None:
            self.groups = [[child] for child in self.children]
        self.num_cuts = len(self.groups)

        if self.is_minimize:
            dummy_bounds = [1e9 for _ in range(self.num_cuts)]  # FIXME
        else:
            dummy_bounds = [-1e9 for _ in range(self.num_cuts)]  # FIXME

        self.alg.build(dummy_bounds)
        self.built = True

    def solve(self) -> None:
        self.alg.solve()

    def get_dual_solution(self) -> List[float]:
        return self.alg.get_solution()

    def add_cuts(self, cuts: Dict[int, Cut]) -> bool:
        aggregate_cuts = []
        assert self.groups is not None
        for group in self.groups:
            group_cut = []
            group_multipliers = []
            for child in group:
                group_cut.append(cuts[child])
                group_multipliers.append(self.children_multipliers[child])
            aggregate_cut = self._aggregate_cuts(group_multipliers, group_cut)

            aggregate_cuts.append(aggregate_cut)
        finished = self.alg.add_cuts(aggregate_cuts)
        return finished

    def _aggregate_cuts(self, multipliers: List[float], cuts: List[Cut]) -> CutList:
        new_coef = [0.0 for _ in range(self.num_constrs)]
        new_constant = 0
        new_objective = 0
        feasibility_cuts = []
        for multiplier, cut in zip(multipliers, cuts):
            if isinstance(cut, OptimalityCut):
                if len(feasibility_cuts) == 0:
                    new_coef = [
                        new_coef[i] + multiplier * cut.coeffs[i]
                        for i in range(self.num_constrs)
                    ]
                    new_constant += multiplier * cut.rhs
                    new_objective += multiplier * cut.objective_value
            elif isinstance(cut, FeasibilityCut):
                feasibility_cuts.append(cut)
        if len(feasibility_cuts) > 0:
            return CutList(feasibility_cuts)
        return CutList(
            [
                OptimalityCut(
                    coeffs=new_coef,
                    rhs=new_constant,
                    objective_value=new_objective,
                )
            ]
        )

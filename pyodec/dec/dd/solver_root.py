from typing import List, Dict

from pyomo.environ import ConcreteModel
from pyomo.core.base.var import VarData

from pyodec.alg.bm.bm import BundleManager
from pyodec.alg.bm.cuts import Cut, OptimalityCut, FeasibilityCut
from .solver import DdSolver


class DdSolverRoot(DdSolver):

    def __init__(self, model: ConcreteModel, solver: str, **kwargs):
        super().__init__(model, solver, **kwargs)

        self.bm = BundleManager(model)

    def build(self, subobj_bounds: List[float]):
        self.bm.build(
            subobj_bounds, self.original_objective, self.get_objective_sense()
        )

    def add_cuts(
        self, iteration: int, cuts_list: List[List[Cut]], vars: List[VarData]
    ) -> List[bool]:
        solution = self.get_solution(vars)
        found_cuts = [False for _ in range(len(cuts_list))]
        for i, cuts in enumerate(cuts_list):
            for cut in cuts:
                found_cut = False
                if isinstance(cut, OptimalityCut):
                    found_cut = self.bm.add_optimality_cut(
                        iteration, i, cut, vars, solution
                    )
                elif isinstance(cut, FeasibilityCut):
                    found_cut = self.bm.add_feasibility_cut(
                        iteration, i, cut, vars, solution
                    )
                found_cuts[i] = found_cut or found_cuts[i]

        return found_cuts

from typing import List

from pyomo.environ import ConcreteModel
from pyomo.core.base.var import VarData

from pyodec.alg.bm.bm import BundleManager
from pyodec.alg.bm.cuts import OptimalityCut, FeasibilityCut, CutList
from .solver import BdSolver


class BdSolverRoot(BdSolver):

    def __init__(self, model: ConcreteModel, solver: str, max_iteration=1000, **kwargs):
        super().__init__(model, solver, **kwargs)

        self.bm = BundleManager(model, max_iteration)

    def build(self, subobj_bounds: List[float]):
        self.bm.build(
            subobj_bounds, self.original_objective, self.get_objective_sense()
        )

    def add_cuts(self, cuts_list: List[CutList], vars: List[VarData]) -> bool:
        solution = self.get_solution(vars)
        found_cuts = [False for _ in range(len(cuts_list))]
        for i, cuts in enumerate(cuts_list):
            for cut in cuts:
                found_cut = False
                if isinstance(cut, OptimalityCut):
                    found_cut = self.bm.add_optimality_cut(i, cut, vars, solution)
                elif isinstance(cut, FeasibilityCut):
                    found_cut = self.bm.add_feasibility_cut(i, cut, vars, solution)
                found_cuts[i] = found_cut or found_cuts[i]

        optimal = not any(found_cuts)
        if optimal:
            self.bm.logger.log_status_optimal()
            return True

        reached_max_iteration = self.bm.increment()
        if reached_max_iteration:
            self.bm.logger.log_status_max_iter()

        return False

    def reset_iteration(self) -> None:
        self.bm.reset_iteration()

from typing import List

from pyomo.environ import ConcreteModel
from pyomo.core.base.var import VarData

from pyodec.alg.bm.bm import BundleMethod
from pyodec.alg.bm.cuts import CutList
from .solver import BdSolver


class BdSolverRoot(BdSolver):

    def __init__(self, model: ConcreteModel, solver: str, max_iteration=1000, **kwargs):
        super().__init__(model, solver, **kwargs)

        self.bm = BundleMethod(model, max_iteration)

    def build(self, subobj_bounds: List[float]):
        self.bm.build(
            subobj_bounds, self.original_objective, self.get_objective_sense()
        )

    def add_cuts(self, cuts_list: List[CutList], vars: List[VarData]) -> bool:
        solution = self.get_solution(vars)
        return self.bm.add_cuts(cuts_list, vars, solution)

    def reset_iteration(self) -> None:
        self.bm.reset_iteration()

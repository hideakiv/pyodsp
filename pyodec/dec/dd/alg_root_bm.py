from typing import List, Dict

from pyomo.environ import ConcreteModel
from pyomo.core.base.var import VarData

from .alg_root import DdAlgRoot
from pyodec.alg.bm.bm import BundleMethod
from pyodec.alg.cuts import CutList


class DdAlgRootBm(DdAlgRoot):

    def __init__(
        self,
        coupling_model: ConcreteModel,
        is_minimize: bool,
        solver_name: str,
        vars_dn: Dict[int, List[VarData]],
        max_iteration=1000,
        **kwargs
    ) -> None:
        super().__init__(coupling_model, is_minimize, solver_name, vars_dn, **kwargs)

        self.bm = BundleMethod(self.solver, max_iteration)
        self.bm.relax_bound.append(None)

    def build(self, subobj_bounds: List[float]) -> None:
        self.bm.build(subobj_bounds)

    def reset_iteration(self) -> None:
        self.bm.reset_iteration(0)

    def solve(self) -> None:
        self.bm.solve()

    def get_solution(self) -> List[float]:
        return self.bm.get_solution()

    def add_cuts(self, cuts_list: List[CutList]) -> bool:
        return self.bm.add_cuts(cuts_list)

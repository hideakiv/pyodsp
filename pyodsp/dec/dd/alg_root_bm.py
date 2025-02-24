from typing import List, Dict
from pathlib import Path

from pyomo.environ import ConcreteModel
from pyomo.core.base.var import VarData

from .alg_root import DdAlgRoot
from pyodsp.alg.bm.bm import BundleMethod
from pyodsp.alg.cuts import CutList
from pyodsp.alg.cuts_manager import CutInfo


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

    def build(self, num_cuts: int) -> None:
        if self.is_minimize:
            dummy_bounds = [1e9 for _ in range(num_cuts)]
        else:
            dummy_bounds = [-1e9 for _ in range(num_cuts)]

        self.bm.build(num_cuts, dummy_bounds)

    def run_step(self, cuts_list: List[CutList] | None) -> List[float] | None:
        return self.bm.run_step(cuts_list)

    def reset_iteration(self) -> None:
        self.bm.reset_iteration()

    def get_cuts(self) -> List[List[CutInfo]]:
        return self.bm.cuts_manager.get_cuts()

    def save(self, dir: Path) -> None:
        self.bm.save(dir)

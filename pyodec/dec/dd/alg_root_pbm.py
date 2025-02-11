from typing import List, Dict
from pathlib import Path

from pyomo.environ import ConcreteModel
from pyomo.core.base.var import VarData

from .alg_root import DdAlgRoot
from pyodec.alg.pbm.pbm import ProximalBundleMethod
from pyodec.alg.cuts import CutList


class DdAlgRootPbm(DdAlgRoot):

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

        self.pbm = ProximalBundleMethod(self.solver, max_iteration)
        self.pbm.obj_bound.append(None)

    def build(self, num_cuts: int) -> None:
        self.pbm.set_init_solution([0.0 for _ in range(num_cuts)])
        self.pbm.build(num_cuts)

    def run_step(self, cuts_list: List[CutList] | None) -> List[float] | None:
        return self.pbm.run_step(cuts_list)

    def reset_iteration(self) -> None:
        self.pbm.reset_iteration()

    def save(self, dir: Path) -> None:
        self.pbm.save(dir)

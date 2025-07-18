from abc import ABC, abstractmethod
from typing import List, Dict

from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    ScalarVar,
)

from pyodsp.alg.cuts_manager import CutInfo
from pyodsp.solver.pyomo_solver import SolverConfig

from .message import DdInitDnMessage, DdFinalUpMessage, DdFinalDnMessage
from ..node._alg import IAlgRoot
from .master_creator import MasterCreator
from .mip_heuristic_root import IMipHeuristicRoot


class DdAlgRoot(IAlgRoot, ABC):
    def __init__(
        self,
        coupling_model: ConcreteModel,
        is_minimize: bool,
        solver_config: SolverConfig,
        vars_dn: Dict[int, List[ScalarVar]],
        heuristic: IMipHeuristicRoot | None = None,
    ) -> None:
        self.coupling_model = coupling_model
        self.vars_dn = vars_dn
        self._init_check()
        mc = MasterCreator(coupling_model, is_minimize, solver_config, vars_dn)
        self.solver = mc.create()
        self.lagrangian_data = mc.lagrangian_data
        self.num_constrs = mc.num_constrs
        self._is_minimize = is_minimize
        self.heuristic = heuristic

        self.is_finalized = False

    def get_vars_dn(self) -> Dict[int, List[ScalarVar]]:
        return self.vars_dn

    def _init_check(self) -> None:
        for obj in self.coupling_model.component_objects(Objective, active=True):
            # There should not be any objective
            raise ValueError("Objective should not be defined in coupling_model")

        # Check that vars_dn is properly specified
        varname_list = []
        for var in self.coupling_model.component_objects(ctype=Var):
            if isinstance(var, ScalarVar):
                varname_list.append(var.name)
            else:
                for index in var:
                    varname_list.append(var[index].name)

        for varlist in self.vars_dn.values():
            for var in varlist:
                if var.name in varname_list:
                    varname_list.pop(varname_list.index(var.name))
                else:
                    raise ValueError(
                        f"Variable {var.name} does not exist in varname_list"
                    )

        if len(varname_list) > 0:
            raise ValueError(f"Variables {varname_list} not coupled")

    def is_minimize(self) -> bool:
        return self._is_minimize

    def get_init_dn_message(self, **kwargs) -> DdInitDnMessage:
        child_id = kwargs["child_id"]
        message = DdInitDnMessage(
            self.lagrangian_data.matrix[child_id], self.is_minimize()
        )
        return message

    def get_coupling_model(self) -> ConcreteModel:
        return self.coupling_model

    @abstractmethod
    def get_final_dn_message(self, **kwargs) -> DdFinalDnMessage:
        if self.heuristic is None:
            return DdFinalDnMessage(None)
        if not self.is_finalized:
            groups = kwargs["groups"]
            self.heuristic.build(
                groups=groups,
                coupling_model=self.coupling_model,
                cuts=self.get_cuts(),
                vars_dn=self.get_vars_dn(),
                is_minimize=self.is_minimize(),
            )
            self.final_solutions = self.heuristic.run()
            self.is_finalized = True

    def pass_final_up_message(self, children_obj: float | None) -> DdFinalUpMessage:
        return DdFinalUpMessage(children_obj)

    @abstractmethod
    def get_cuts(self) -> List[List[CutInfo]]:
        pass

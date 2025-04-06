from abc import ABC, abstractmethod
from typing import List, Dict

from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    ScalarVar,
)
from pyomo.core.base.var import IndexedVar

from pyodsp.alg.cuts_manager import CutInfo
from pyodsp.dec.run._message import DdInitMessage

from ..node._alg import IAlgRoot
from .master_creator import MasterCreator

class DdAlgRoot(IAlgRoot, ABC):

    def __init__(
        self,
        coupling_model: ConcreteModel,
        is_minimize: bool,
        solver_name: str,
        vars_dn: Dict[int, List[ScalarVar]],
        **kwargs
    ) -> None:
        self.coupling_model = coupling_model
        self.vars_dn = vars_dn
        self._init_check()
        mc = MasterCreator(
            coupling_model, is_minimize, solver_name, vars_dn, **kwargs
        )
        self.solver = mc.create()
        self.lagrangian_data = mc.lagrangian_data
        self.num_constrs = mc.num_constrs
        self._is_minimize = is_minimize

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
            elif isinstance(var, IndexedVar):
                for index in var:
                    varname_list.append(var[index].name)

        for varlist in self.vars_dn.values():
            for var in varlist:
                if var.name in varname_list:
                    varname_list.pop(varname_list.index(var.name))
                else:
                    raise ValueError(f"Variable {var.name} does not exist in varname_list")
        
        if len(varname_list) > 0:
            raise ValueError(f"Variables {varname_list} not coupled")
        
    def is_minimize(self) -> bool:
        return self._is_minimize
    
    def get_init_message(self, **kwargs) -> DdInitMessage:
        child_id = kwargs["child_id"]
        message = DdInitMessage(self.lagrangian_data.matrix[child_id])
        return message

    @abstractmethod
    def get_cuts(self) -> List[List[CutInfo]]:
        pass

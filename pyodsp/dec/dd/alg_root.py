from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    RangeSet,
    Objective,
    minimize,
    maximize,
    NonNegativeReals,
    Reals,
)
from pyomo.core.base.var import VarData, IndexedVar

from pyodsp.dec.utils import get_nonzero_coefficients_group
from pyodsp.solver.pyomo_solver import PyomoSolver
from pyodsp.alg.cuts import CutList
from pyodsp.alg.cuts_manager import CutInfo
from pyodsp.alg.params import DEC_CUT_ABS_TOL

from ..node._alg import IAlgRoot

class DdAlgRoot(IAlgRoot, ABC):

    def __init__(
        self,
        coupling_model: ConcreteModel,
        is_minimize: bool,
        solver_name: str,
        vars_dn: Dict[int, List[VarData]],
        **kwargs
    ) -> None:
        self.coupling_model = coupling_model
        self.vars_dn = vars_dn
        self._init_check()
        self.solver = self._create_master(
            coupling_model, is_minimize, solver_name, vars_dn, **kwargs
        )
        self._is_minimize = is_minimize

    def _create_master(
        self,
        coupling_model: ConcreteModel,
        is_minimize: bool,
        solver_name: str,
        vars_dn: Dict[int, List[VarData]],
        **kwargs
    ) -> PyomoSolver:
        master: ConcreteModel = ConcreteModel()
        self.lagrangian_data = get_nonzero_coefficients_group(coupling_model, vars_dn)
        self.num_constrs = len(self.lagrangian_data.constraints)

        master.ld_plus = Var(
            RangeSet(0, self.num_constrs - 1), domain=NonNegativeReals, initialize=0
        )
        master.ld_minus = Var(
            RangeSet(0, self.num_constrs - 1), domain=NonNegativeReals, initialize=0
        )
        for i in range(self.num_constrs):
            if self.lagrangian_data.lbs[i] is None:
                master.ld_minus[i].fix(0)
            if self.lagrangian_data.ubs[i] is None:
                master.ld_plus[i].fix(0)
        master.ld = Var(RangeSet(0, self.num_constrs - 1), domain=Reals, initialize=0)

        def constr_rule(m, i):
            return m.ld[i] == m.ld_plus[i] - m.ld_minus[i]

        master.constr = Constraint(RangeSet(0, self.num_constrs - 1), rule=constr_rule)

        def min_obj(m):
            expr = 0.0
            for i in range(self.num_constrs):
                ub = self.lagrangian_data.ubs[i]
                if ub is not None and abs(ub) > DEC_CUT_ABS_TOL:
                    expr -= ub * m.ld_plus[i]
                lb = self.lagrangian_data.lbs[i]
                if lb is not None and abs(lb) > DEC_CUT_ABS_TOL:
                    expr += lb * m.ld_minus[i]
            return expr

        def max_obj(m):
            return -min_obj(m)

        if is_minimize:
            master.objective = Objective(rule=min_obj, sense=maximize)
        else:
            master.objective = Objective(rule=max_obj, sense=minimize)

        lagrangian_duals: List[VarData] = [
            master.ld[i] for i in range(self.num_constrs)
        ]
        return PyomoSolver(master, solver_name, lagrangian_duals, **kwargs)

    def get_vars_dn(self) -> Dict[int, List[VarData]]:
        return self.vars_dn

    def _init_check(self) -> None:
        for obj in self.coupling_model.component_objects(Objective, active=True):
            # There should not be any objective
            raise ValueError("Objective should not be defined in coupling_model")
        
        # Check that vars_dn is properly specified
        varname_list = []
        for var in self.coupling_model.component_objects(ctype=Var):
            if isinstance(var, VarData):
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

    @abstractmethod
    def build(self, num_cuts: int) -> None:
        pass

    @abstractmethod
    def run_step(self, cuts_list: List[CutList] | None) -> Tuple[int, List[float]]:
        pass

    @abstractmethod
    def reset_iteration(self) -> None:
        pass

    @abstractmethod
    def get_cuts(self) -> List[List[CutInfo]]:
        pass

    @abstractmethod
    def set_logger(self, node_id: int, depth: int) -> None:
        pass

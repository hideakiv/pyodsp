from abc import abstractmethod
from typing import List, Dict

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
from pyomo.core.base.var import VarData

from pyodec.dec.utils import get_nonzero_coefficients_group
from pyodec.solver.pyomo_solver import PyomoSolver
from pyodec.alg.cuts import CutList


class DdAlgRoot:

    def __init__(
        self,
        coupling_model: ConcreteModel,
        is_minimize: bool,
        solver_name: str,
        vars_dn: Dict[int, List[VarData]],
        **kwargs
    ) -> None:
        self.solver = self._create_master(
            coupling_model, is_minimize, solver_name, vars_dn, **kwargs
        )
        self.vars_dn = vars_dn
        self.is_minimize = is_minimize

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
                if ub is not None:
                    expr -= ub * m.ld_plus[i]
                lb = self.lagrangian_data.lbs[i]
                if lb is not None:
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

    @abstractmethod
    def build(self, num_cuts: int) -> None:
        pass

    @abstractmethod
    def run_step(self, cuts_list: List[CutList] | None) -> List[float] | None:
        pass

    @abstractmethod
    def reset_iteration(self) -> None:
        pass

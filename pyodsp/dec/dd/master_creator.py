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
    ScalarVar,
)

from pyodsp.dec.utils import get_nonzero_coefficients_group
from pyodsp.solver.pyomo_solver import PyomoSolver
from pyodsp.alg.params import DEC_CUT_ABS_TOL


class MasterCreator:
    def __init__(
        self,
        coupling_model: ConcreteModel,
        is_minimize: bool,
        solver_name: str,
        vars_dn: Dict[int, List[ScalarVar]],
        **kwargs,
    ) -> None:
        self.lagrangian_data = get_nonzero_coefficients_group(coupling_model, vars_dn)
        self.num_constrs = len(self.lagrangian_data.constraints)
        self.is_minimize = is_minimize
        self.solver_name = solver_name
        self.kwargs = kwargs

    def create(self) -> PyomoSolver:
        master: ConcreteModel = ConcreteModel()

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

        if self.is_minimize:
            master.objective = Objective(rule=min_obj, sense=maximize)
        else:
            master.objective = Objective(rule=max_obj, sense=minimize)

        lagrangian_duals: List[ScalarVar] = [
            master.ld[i] for i in range(self.num_constrs)
        ]
        return PyomoSolver(master, self.solver_name, lagrangian_duals, **self.kwargs)

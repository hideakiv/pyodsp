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
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.alg.params import DEC_CUT_ABS_TOL, BM_LAMBDA_BOUND


class MasterCreator:
    def __init__(
        self,
        is_minimize: bool,
        solver_config: SolverConfig,
    ) -> None:
        self.is_minimize = is_minimize
        self.solver_config = solver_config

    def create(self, solution: List[float], rho: float) -> PyomoSolver:
        master: ConcreteModel = ConcreteModel()

        # alpha is actually _theta in BundleMethod so we do not add it here.

        master.beta = Var(
            range(len(solution)),
            domain=Reals,
        )
        master.tau = Var(
            domain=NonNegativeReals,
        )

        def min_obj(m):
            expr = -rho * (1 + master.tau)
            for i, sol in enumerate(solution):
                expr -= master.beta[i] * sol
            return expr

        def max_obj(m):
            raise NotImplementedError("Maximization not implemented yet")
            return -min_obj(m)

        if self.is_minimize:
            master.objective = Objective(rule=min_obj, sense=maximize)
        else:
            master.objective = Objective(rule=max_obj, sense=minimize)

        vars = [master.tau] + [master.beta[i] for i in range(len(solution))]

        return PyomoSolver(master, self.solver_config, vars)

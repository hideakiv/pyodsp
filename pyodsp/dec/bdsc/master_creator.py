from typing import List, Dict

from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    RangeSet,
    Objective,
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
        solver_config: SolverConfig,
    ) -> None:
        self.solver_config = solver_config

    def create(
        self, solution: List[float], rho: float, coef_bound: float | None = None
    ) -> PyomoSolver:
        """The cut-generation master (CGMP) of van der Laan & Romeijnders,
        Section 4.1, at trial point `solution` and penalty `rho`.

        coef_bound caps |beta| and tau, as the paper does at 1e8 to keep
        the cut coefficients numerically sane. It is off by default: the
        first column already bounds the objective, and a box would make
        that first, degenerate LP's simplex solution sit on the box.
        """
        master: ConcreteModel = ConcreteModel()

        # alpha is actually _theta in BundleMethod so we do not add it here.

        beta_bounds = (None, None) if coef_bound is None else (-coef_bound, coef_bound)
        master.beta = Var(
            range(len(solution)),
            domain=Reals,
            bounds=beta_bounds,
        )
        master.tau = Var(
            domain=NonNegativeReals,
            bounds=(0.0, coef_bound),
        )

        def min_obj(m):
            expr = -rho * (1 + master.tau)
            for i, sol in enumerate(solution):
                expr -= master.beta[i] * sol
            return expr

        # Pricing is the dual of the restricted master, so this problem is
        # deliberately a maximize one; it is exempt from PyomoSolver's
        # maximize-to-minimize conversion for the same reason dual
        # decomposition's Lagrangian master is.
        master.objective = Objective(rule=min_obj, sense=maximize)

        vars = [master.tau] + [master.beta[i] for i in range(len(solution))]

        return PyomoSolver(master, self.solver_config, vars, convert_maximize=False)

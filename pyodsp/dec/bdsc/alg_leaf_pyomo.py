from typing import List
from pathlib import Path
import time
import pandas as pd
import logging

from pyomo.environ import value

from .master_creator import MasterCreator
from .message import (
    BdScInitDnMessage,
    BdScInitUpMessage,
    BdScFinalDnMessage,
    BdScFinalUpMessage,
    BdScDnMessage,
    BdScUpMessage,
)
from ..node._alg import IAlgLeaf
from pyodsp.solver.pyomo_utils import update_linear_terms_in_objective
from pyodsp.alg.bm.bm import BundleMethod
from pyodsp.alg.bm.pbm import ProximalBundleMethod
from pyodsp.alg.bm.cuts import Cut, OptimalityCut, FeasibilityCut, CutList
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.alg.params import DEC_CUT_ABS_TOL
from pyodsp.alg.const import STATUS_NOT_FINISHED


class BdScAlgLeafPyomo(IAlgLeaf):
    def __init__(
        self, solver: PyomoSolver, master_config: SolverConfig, max_iteration=1000
    ):
        self.cgsp = BundleMethod(solver, max_iteration)
        self.mc = MasterCreator(
            is_minimize=self.cgsp.is_minimize(), solver_config=master_config
        )
        self.max_iteration = max_iteration
        self.step_time: List[float] = []

    def build(self) -> None:
        self.cgsp.set_logger(
            node_id=0, depth=1, level=logging.DEBUG
        )  # TODO: pass actual node id and depth
        if self.cgsp.is_minimize():
            self.cgsp.build(1, [-1e9])  # TODO: use a better bound
        else:
            raise ValueError("Maximization is not supported yet")
            self.cgsp.build(1, [1e9])

    def pass_init_dn_message(self, message: BdScInitDnMessage) -> None:
        if self.is_minimize() != message.get_is_minimize():
            raise ValueError("Inconsistent optimization sense")

    def get_init_up_message(self) -> BdScInitUpMessage:
        return BdScInitUpMessage()

    def pass_dn_message(self, message: BdScDnMessage) -> None:
        solution = message.get_solution()
        rho = message.get_rho()
        objective = message.get_objective()
        cut_list = message.get_cut()
        if cut_list is None:
            # self._fix_variables(solution)
            # self._fix_parent_objective(objective)
            master = self.mc.create(
                solution, rho
            )  # NOTE: maybe we can reuse the master from the previous iteration
            self.cgmp = ProximalBundleMethod(master, self.max_iteration)
            self.cgmp.set_logger(
                node_id=0, depth=1, level=logging.DEBUG
            )  # TODO: pass actual node id and depth
            init_solution = [0.0 for _ in range(len(solution) + 1)]
            self.cgmp.set_init_solution(init_solution)
            self.cgmp.build(1)
            self.cgsp.reset_iteration()
        else:
            self.cgsp.add_cuts(cut_list)

    def pass_final_dn_message(self, message: BdScFinalDnMessage) -> None:
        pass

    def get_final_up_message(self) -> BdScFinalUpMessage:
        return BdScFinalUpMessage(self.cgsp.cpm.solver.get_original_objective_value())

    def _update_cgsp_objective(
        self, alpha: float, beta: list[float], tau: float
    ) -> None:
        vars = [var for var in self.cgsp.get_vars()]  # for beta
        vars.append(self.cgsp.cpm.solver.model._theta[0])  # for tau
        vars.append(-1.0)  # for alpha
        coeffs = [b for b in beta]
        coeffs.append(tau)
        coeffs.append(alpha)
        update_linear_terms_in_objective(self.cgsp.cpm.solver, coeffs, vars)

    def _fix_variables(self, coupling_values: List[float]) -> None:
        """Fix the variables to a specified value

        Args:
            vars: The variables to be fixed.
            values: The values to be set.
        """
        for i, var in enumerate(self.cgsp.cpm.solver.vars):
            var.fix(coupling_values[i])

    def _unfix_variables(self) -> None:
        for var in self.cgsp.cpm.solver.vars:
            var.unfix()

    def _fix_parent_objective(self, objective: float) -> None:
        self.cgsp.cpm.solver.set_parent_objective_value(objective)

    def get_up_message(self) -> BdScUpMessage:
        start = time.time()
        cuts_list = None
        self.cgsp.cpm.solver.original_objective.deactivate()
        for _ in range(self.max_iteration):
            status, solution, objective = self.cgmp.run_step(cuts_list)
            if status != STATUS_NOT_FINISHED:
                break
            alpha = self.cgmp.cpm.get_theta_value()[0]
            tau = solution[0]
            beta = solution[1:]
            self._update_cgsp_objective(alpha, beta, tau)
            self.cgsp.cpm.solve()
            qy = self.cgsp.get_original_objective_value()
            x = [value(var) for var in self.cgsp.get_vars()]
            theta = self.cgsp.cpm.get_theta_value()[0]
            coeffs = {0: -theta}
            for i, val in enumerate(x):
                coeffs[i + 1] = -val
            cgcut = OptimalityCut(
                coeffs=coeffs,
                rhs=qy,
                objective_value=None,
                info={},
            )
            cuts_list = [CutList([cgcut])]
        self.step_time.append(time.time() - start)

        c = self.cgmp.cpm.get_objective_value()
        alpha = self.cgmp.cpm.get_theta_value()[0]
        tau = solution[0]
        beta = solution[1:]
        coeffs = {}
        for i, val in enumerate(beta):
            coeffs[i] = val
        cut = OptimalityCut(
            coeffs=coeffs,
            rhs=alpha,
            objective_value=None,
            info={},
        )
        return BdScUpMessage(cut, c, tau, None)

    def save(self, dir: Path) -> None:
        self.cgmp.save(dir)
        path = dir / "step_time.csv"
        df = pd.DataFrame(self.step_time, columns=["step_time"])
        df.to_csv(path, index=False)

    def is_minimize(self) -> bool:
        return self.cgsp.is_minimize()

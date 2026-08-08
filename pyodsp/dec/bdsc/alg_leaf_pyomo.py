from typing import List
from pathlib import Path
import time
import pandas as pd
import logging

from pyomo.environ import value, Constraint

from .master_creator import MasterCreator
from .message import (
    BdScInitDnMessage,
    BdScInitUpMessage,
    BdScFinalDnMessage,
    BdScFinalUpMessage,
    BdScDnMessage,
    BdScUpMessage,
)
from ..node._message import NodeIdx
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
        if not solver.is_minimize():
            raise ValueError(
                "Benders decomposition with scaled cuts only accepts minimize "
                "problems; negate the objective (see "
                "pyomo_utils.negate_objective_sense) before constructing the "
                "solver."
            )
        # The column-generation subproblem must price against exactly the
        # master's cuts, so it makes no add/drop decisions of its own:
        # force=True bypasses its dominance check and purgeable=False stops
        # it aging cuts out on its own schedule. Both would otherwise be
        # decided from this subproblem's solves, which are not the master's.
        # It still drops cuts when told to — see pass_dn_message.
        self.cgsp = BundleMethod(solver, max_iteration, force=True, purgeable=False)
        self.mc = MasterCreator(
            is_minimize=self.cgsp.is_minimize(), solver_config=master_config
        )
        self.max_iteration = max_iteration
        self.step_time: List[float] = []
        self.cgmp_id = 0

    def build(self) -> None:
        self.cgsp.set_logger(
            node_id=f"{self.idx}_cgsp", depth=1, level=self.level
        )  # TODO: pass actual node id and depth
        self.cgsp.build(1, [-1e9])  # temporary bound

    def pass_init_dn_message(self, message: BdScInitDnMessage) -> None:
        if self.is_minimize() != message.get_is_minimize():
            raise ValueError("Inconsistent optimization sense")

    def get_init_up_message(self) -> BdScInitUpMessage:
        return BdScInitUpMessage()

    def pass_dn_message(self, message: BdScDnMessage) -> None:
        solution = message.get_solution()
        rho = message.get_rho()
        objective = message.get_objective()
        cut_list = message.get_cut_list()
        subobj_bound = sum(message.get_subobj_bounds())  # TODO: update only once
        self.cgsp.cpm.solver.model._theta[0].setlb(subobj_bound)

        self._fix_variables(solution)
        self._fix_parent_objective(objective)
        self._create_master(solution, rho)
        if cut_list is not None:
            # the message carries the master's whole cut set, so mirror it
            # wholesale rather than appending — that is what keeps this
            # subproblem's cuts identical to the master's across the
            # master's own drops and refusals
            if len(cut_list) != self.cgsp.num_cuts:
                raise ValueError(
                    f"Master sent {len(cut_list)} cut group(s), but this "
                    f"subproblem prices {self.cgsp.num_cuts}; Benders "
                    "decomposition with scaled cuts needs the root's children "
                    "in a single group (see DecNodeParent.set_groups)"
                )
            self.cgsp.replace_cuts(cut_list)
        self.cgsp.reset_iteration()

    def _create_master(self, solution: List[float], rho: float) -> None:
        master = self.mc.create(
            solution, rho
        )  # NOTE: maybe we can reuse the master from the previous iteration
        self.cgmp = ProximalBundleMethod(master, self.max_iteration)
        self.cgmp.set_logger(
            node_id=f"{self.idx}_cgmp_{self.cgmp_id}", depth=1, level=self.level
        )
        self.cgmp_id += 1
        init_solution = [0.0 for _ in range(len(solution) + 1)]
        self.cgmp.set_init_solution(init_solution)
        self.cgmp.build(1)

    def pass_final_dn_message(self, message: BdScFinalDnMessage) -> None:
        pass

    def get_final_up_message(self) -> BdScFinalUpMessage:
        return BdScFinalUpMessage(self.cgsp.cpm.solver.get_original_objective_value())

    def _update_cgsp_objective(self, beta: list[float], tau: float) -> None:
        # ignore alpha
        vars = [var for var in self.cgsp.get_vars()]  # for beta
        vars.append(self.cgsp.cpm.solver.model._theta[0])  # for tau
        coeffs = [b for b in beta]
        coeffs.append(tau)
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
        self.cgsp.cpm.solver.activate_original_objective()
        self.cgsp.cpm.solver.solve()
        leaf_objective = self.cgsp.cpm.solver.get_objective_value()
        self.cgsp.cpm.solver.original_objective.deactivate()
        self._unfix_variables()
        for _ in range(self.max_iteration):
            if _ > 0:
                # print("cgsp", cuts_list[0][0])
                # breakpoint()
                status, solution, objective = self.cgmp.run_step(cuts_list)
                if status != STATUS_NOT_FINISHED:
                    break
                tau = solution[0]
                beta = solution[1:]
            else:
                tau = 0.0
                beta = [0.0 for var in self.cgsp.get_vars()]
            self._update_cgsp_objective(beta, tau)
            # print("cgmp", beta, tau)
            # breakpoint()
            self.cgsp.cpm.solve()
            qy = self.cgsp.get_original_objective_value()
            x = [value(var) for var in self.cgsp.get_vars()]
            theta = self.cgsp.cpm.get_theta_value(0)
            coeffs = {0: -theta}
            for i, val in enumerate(x):
                coeffs[i + 1] = -val
            cgcut = OptimalityCut(
                coeffs=coeffs,
                rhs=qy,
                objective_value=self.cgsp.get_objective_value(),
                info={},
            )
            cuts_list = [CutList([cgcut])]
        self.step_time.append(time.time() - start)

        c = self.cgmp.cpm.get_objective_value()
        alpha = self.cgmp.cpm.get_theta_value(0)
        tau = solution[0]
        beta = solution[1:]
        coeffs = {}
        for i, val in enumerate(beta):
            coeffs[i] = val
        cut = OptimalityCut(
            coeffs=coeffs,
            rhs=alpha,
            objective_value=leaf_objective,
            info={},
        )
        root_objective = self.cgsp.cpm.get_parent_objective_value()
        return BdScUpMessage(cut, c, tau, root_objective + leaf_objective)

    def set_logger(self, idx: NodeIdx, depth: int, level: int) -> None:
        self.idx = idx
        self.depth = depth
        self.level = level

    def save(self, dir: Path) -> None:
        self.cgmp.save(dir)
        path = dir / "step_time.csv"
        df = pd.DataFrame(self.step_time, columns=["step_time"])
        df.to_csv(path, index=False)

    def is_minimize(self) -> bool:
        return self.cgsp.is_minimize()

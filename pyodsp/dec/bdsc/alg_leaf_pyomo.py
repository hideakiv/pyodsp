import logging
import time
from pathlib import Path
from typing import List

import pandas as pd
from pyomo.environ import Constraint, value

from pyodsp.alg.bm.bm import BundleMethod
from pyodsp.alg.bm.cuts import Cut, CutList, FeasibilityCut, OptimalityCut
from pyodsp.alg.bm.pbm import ProximalBundleMethod
from pyodsp.alg.const import STATUS_NOT_FINISHED
from pyodsp.alg.params import DEC_CUT_ABS_TOL
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.solver.pyomo_utils import update_linear_terms_in_objective

from ..node._alg import IAlgLeaf
from ..node._message import NodeIdx
from .master_creator import MasterCreator
from .message import (
    BdScDnMessage,
    BdScFinalDnMessage,
    BdScFinalUpMessage,
    BdScInitDnMessage,
    BdScInitUpMessage,
    BdScUpMessage,
)


class BdScAlgLeafPyomo(IAlgLeaf):
    def __init__(
        self, solver: PyomoSolver, master_config: SolverConfig, max_iteration=1000
    ):
        if not solver.is_minimize():
            raise ValueError(
                "Benders decomposition with scaled cuts needs a minimize model. PyomoSolver converts a "
                "maximize one on construction, so this solver was built "
                "with convert_maximize=False — that is reserved for the "
                "internal masters whose sense is deliberately inverted."
            )
        # The column-generation subproblem must price against exactly the
        # master's cuts, so it makes no add/drop decisions of its own:
        # force=True bypasses its dominance check and purgeable=False stops
        # it aging cuts out on its own schedule. Both would otherwise be
        # decided from this subproblem's solves, which are not the master's.
        # It still drops cuts when told to — see pass_dn_message.
        self.cgsp = BundleMethod(solver, max_iteration, force=True, purgeable=False)
        self.mc = MasterCreator(solver_config=master_config)
        self.max_iteration = max_iteration
        self.step_time: List[float] = []
        # columns carried over from the previous trial point — see _sync_cuts
        self._cgmp_cuts: List[Cut] = []
        self._subobj_bound: float | None = None

    def build(self) -> None:
        # this node's own depth: the cgsp and cgmp are its two inner solvers,
        # not nodes of their own, so they sit at the depth of the leaf that
        # runs them and are told apart by the suffix on the id
        self.cgsp.set_logger(
            node_id=f"{self.idx}_cgsp", depth=self.depth, level=self.level
        )
        # A placeholder floor under theta: the real one is the root's, and it
        # cannot be known here yet — see _set_subobj_bound, which installs it
        # when the first trial point arrives.
        self.cgsp.build(1, [-1e9])

    def pass_init_dn_message(self, message: BdScInitDnMessage) -> None:
        pass

    def get_init_up_message(self) -> BdScInitUpMessage:
        return BdScInitUpMessage()

    def pass_dn_message(self, message: BdScDnMessage) -> None:
        solution = message.get_solution()
        rho = message.get_rho()
        objective = message.get_objective()
        cut_list = message.get_cut_list()

        self._set_subobj_bound(message.get_subobj_bounds())
        self._sync_cuts(cut_list)
        self._fix_variables(solution)
        self._fix_parent_objective(objective)
        self._create_master(solution, rho)
        self.cgsp.reset_iteration()

    def _set_subobj_bound(self, subobj_bounds: List[float | None]) -> None:
        """Install the floor under the cgsp's theta. Only the first call does.

        The root works these bounds out in its own build(), from bounds its
        children report, and they do not change afterwards — but it has no
        way to hand them over any earlier than this: at init time those
        bounds are still travelling *up* from the leaves, so
        BdScInitDnMessage cannot carry them. They ride along on every dn
        message instead, and the first to arrive is the one that counts. A
        later change is refused rather than applied: raising this floor
        shrinks the cgsp, which would quietly invalidate every column the
        cgmp has collected since (see _sync_cuts).
        """
        if len(subobj_bounds) != self.cgsp.num_cuts:
            raise ValueError(
                f"Master sent {len(subobj_bounds)} subproblem bound(s), but "
                f"this subproblem prices {self.cgsp.num_cuts}; Benders "
                "decomposition with scaled cuts needs the root's children in "
                "a single group (see DecNodeParent.set_groups)"
            )
        if any(bound is None for bound in subobj_bounds):
            raise ValueError(
                "Benders decomposition with scaled cuts needs a bound on every "
                "child's objective to put a floor under the column-generation "
                "subproblem; call set_bound on each leaf node"
            )

        bound = sum(subobj_bounds)
        if self._subobj_bound is None:
            self._subobj_bound = bound
            self.cgsp.cpm.solver.model._theta[0].setlb(bound)
        elif abs(bound - self._subobj_bound) > DEC_CUT_ABS_TOL:
            raise ValueError(
                f"The subproblem bound changed from {self._subobj_bound} to "
                f"{bound}; it is read once, from the first trial point to "
                "arrive, and taken as fixed from then on"
            )

    def _sync_cuts(self, cut_list: list[CutList] | None) -> None:
        """Mirror the master's cuts, and decide whether the columns the cgmp
        has already generated survive into the next one.

        Each cgmp cut records a column — a point (x, theta) the cgsp
        produced — and bounds alpha below by that column's value. It stays
        valid exactly as long as that column stays feasible for the cgsp, so
        a cut the master *added* retires the collected columns: the cgsp
        shrinks under it. Anything that relaxes the cgsp (a cut the master
        purged) leaves them valid, and so does a new trial point on its own
        — y and rho appear only in the cgmp's objective (see
        MasterCreator.create), never in these cuts. The only other thing
        that could shrink the cgsp, the floor under theta, is fixed for the
        run (see _set_subobj_bound).
        """
        if cut_list is None:
            # the master's cuts are unchanged, so every column still holds
            return

        # the message carries the master's whole cut set, so mirror it
        # wholesale rather than appending — that is what keeps this
        # subproblem's cuts identical to the master's across the master's
        # own drops and refusals
        if len(cut_list) != self.cgsp.num_cuts:
            raise ValueError(
                f"Master sent {len(cut_list)} cut group(s), but this "
                f"subproblem prices {self.cgsp.num_cuts}; Benders "
                "decomposition with scaled cuts needs the root's children "
                "in a single group (see DecNodeParent.set_groups)"
            )
        held = self._cut_keys(c.cut for group in self.cgsp.get_cuts() for c in group)
        incoming = self._cut_keys(cut for group in cut_list for cut in group)
        if incoming - held:
            self._cgmp_cuts = []
        self.cgsp.replace_cuts(cut_list)

    @staticmethod
    def _cut_keys(cuts) -> set:
        return {
            (type(cut).__name__, cut.rhs, tuple(sorted(cut.coeffs.items())))
            for cut in cuts
        }

    def _create_master(self, solution: List[float], rho: float) -> None:
        master = self.mc.create(solution, rho)
        self.cgmp = ProximalBundleMethod(master, self.max_iteration)
        # a stable id: successive masters are the same component at
        # successive trial points, so they share one logger rather than
        # leaving a new one behind on every call
        self.cgmp.set_logger(
            node_id=f"{self.idx}_cgmp", depth=self.depth, level=self.level
        )
        init_solution = [0.0 for _ in range(len(solution) + 1)]
        self.cgmp.set_init_solution(init_solution)
        self.cgmp.build(1)
        if self._cgmp_cuts:
            # start from the columns that are still valid rather than making
            # column generation rediscover them, each at the price of a cgsp
            # solve. The master itself is rebuilt because its objective is a
            # function of the new y and rho.
            self.cgmp.add_cuts([CutList(list(self._cgmp_cuts))])

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
                status, solution, objective = self.cgmp.run_step(cuts_list)
                if status != STATUS_NOT_FINISHED:
                    break
                tau = solution[0]
                beta = solution[1:]
            else:
                tau = 0.0
                beta = [0.0 for var in self.cgsp.get_vars()]
            self._update_cgsp_objective(beta, tau)
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

        # carry the master's surviving cuts (its own dominance and aging
        # decisions already applied) into the next trial point
        self._cgmp_cuts = [c.cut for group in self.cgmp.cpm.get_cuts() for c in group]

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

    def get_sense_multiplier(self) -> float:
        return self.cgsp.get_solver().sense_multiplier

    def is_minimize(self) -> bool:
        # Always: PyomoSolver converts a maximize model on construction.
        return True

import logging
import time
from pathlib import Path
from typing import List

import pandas as pd
from pyomo.environ import Constraint, value

from pyodsp.alg.bm.bm import BundleMethod
from pyodsp.alg.bm.cuts import Cut, CutList, FeasibilityCut, OptimalityCut
from pyodsp.alg.bm.pbm import ProximalBundleMethod
from pyodsp.alg.const import (
    STATUS_INFEASIBLE,
    STATUS_MAX_ITERATION,
    STATUS_NOT_FINISHED,
    STATUS_TIME_LIMIT,
)
from pyodsp.alg.params import BM_ABS_TOLERANCE, BM_DUMMY_BOUND, DEC_CUT_ABS_TOL
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
        self,
        solver: PyomoSolver,
        master_config: SolverConfig,
        max_iteration=1000,
        cut_master: str = "bm",
        tolerance: float = BM_ABS_TOLERANCE,
        coef_bound: float | None = None,
    ):
        """
        Args:
            solver: This scenario's model, first stage embedded.
            master_config: Solver for the 'pbm' cut master, which is
                quadratic. The 'bm' master is an LP and is solved with
                `solver`'s own solver.
            max_iteration: Cap on the column generation per trial point.
            cut_master: How the cut-generation master is solved. 'bm' is
                the row generation of van der Laan & Romeijnders
                (Algorithm 2) on a plain cutting-plane master; 'pbm' is the
                earlier proximal bundle variant, kept for comparison.
            tolerance: delta in Algorithm 2 — stop once the cut is within
                this of the best one for the current rho ('bm' only).
            coef_bound: Optional cap on |beta| and tau ('bm' only).
        """
        if cut_master not in ("bm", "pbm"):
            raise ValueError(f"cut_master must be 'bm' or 'pbm', got {cut_master!r}")
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
        self.cut_master = cut_master
        self.tolerance = tolerance
        self.coef_bound = coef_bound
        self.mc = MasterCreator(
            solver_config=master_config if cut_master == "pbm" else solver.solver_config
        )
        self.max_iteration = max_iteration
        self.step_time: List[float] = []
        # columns carried over from the previous trial point — see _sync_cuts
        self._cgmp_cuts: List[Cut] = []
        self._subobj_bound: float | None = None
        self._final_objective: float | None = None

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
        self._x_hat = list(solution)
        self._rho = rho
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
        if self.cut_master == "bm":
            self._create_row_generation_master(solution, rho)
            return
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

    def _create_row_generation_master(self, solution: List[float], rho: float) -> None:
        """The CGMP as a plain cutting-plane master.

        Its columns are added as cuts on alpha (the BundleMethod's theta):
        a column (x_k, theta_k, q_k) is alpha <= q_k + beta^T x_k +
        tau*theta_k. They are never aged out — dropping one only loosens
        the LP and makes column generation find it again. The columns
        carried over from the previous trial point are added here; the
        first column of this one is added by get_up_message, which is
        where y-bar becomes known.
        """
        master = self.mc.create(solution, rho, coef_bound=self.coef_bound)
        self.cgmp = BundleMethod(master, self.max_iteration, purgeable=False)
        self.cgmp.set_logger(
            node_id=f"{self.idx}_cgmp", depth=self.depth, level=self.level
        )
        # BundleMethod's bookkeeping wants a number for alpha's bound, but the
        # variable itself must stay free: the first column leaves the LP
        # optimal along alpha - beta^T x_hat = const, and a ceiling on alpha
        # gives the simplex a vertex at the far end of that ray (alpha at
        # the ceiling, beta ~ ceiling / x_hat), i.e. a useless cut. Free,
        # alpha and beta stay where the simplex leaves free variables.
        self.cgmp.build(1, [BM_DUMMY_BOUND])
        if self.coef_bound is None:
            master.model._theta[0].setlb(None)
        else:
            master.model._theta[0].setlb(-self.coef_bound)
        if self._cgmp_cuts:
            self.cgmp.add_cuts([CutList(list(self._cgmp_cuts))])

    @staticmethod
    def _column(
        x: List[float], theta: float, q: float, value_now: float
    ) -> OptimalityCut:
        """A point (x, theta, y) of S^phi as a cgmp constraint:
        alpha <= q(y) + beta^T x + tau * theta.

        value_now is that right-hand side at the cgmp's current (beta,
        tau); the cgmp declines the column when alpha already satisfies it.
        """
        coeffs = {0: -theta}
        for i, val in enumerate(x):
            coeffs[i + 1] = -val
        return OptimalityCut(coeffs=coeffs, rhs=q, objective_value=value_now, info={})

    def pass_final_dn_message(self, message: BdScFinalDnMessage) -> None:
        """Evaluate the recourse at the master's final solution.

        The cgsp's last solve cannot stand in for it: column generation
        runs with the coupling variables unfixed (see get_up_message), so
        its y answers some x the cgsp chose, not the master's. Paired with
        the master's x that y can be infeasible, and the objective reported
        from it then undercuts the optimum. Fixing x and re-solving the
        original problem gives the cost this x actually incurs.
        """
        solution = message.get_solution()
        assert solution is not None
        solver = self.cgsp.cpm.solver
        self._fix_variables(solution)
        solver.activate_original_objective()
        solver.solve()
        self._final_objective = (
            solver.get_original_objective_value() if solver.is_optimal() else None
        )
        solver.original_objective.deactivate()
        self._unfix_variables()

    def get_final_up_message(self) -> BdScFinalUpMessage:
        return BdScFinalUpMessage(self._final_objective)

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
        if self.cut_master == "bm":
            return self._get_up_message_row_generation()
        return self._get_up_message_pbm()

    def _get_up_message_row_generation(self) -> BdScUpMessage:
        """C_s(rho) by row generation (van der Laan & Romeijnders, Alg. 2).

        Each round solves the CGMP for (alpha, beta, tau), then the CGSP
        min{q(y) + beta^T x + tau*theta : (x, theta, y) in S^phi} with x
        free. The CGSP's value m is what alpha may be for (beta, tau) to
        give a valid cut, so (m, beta, tau) is a feasible point of (19)
        with value m - beta^T x_hat - rho(1 + tau) — a lower bound on C_s.
        The CGMP's value is an upper bound. The round stops once they are
        within `tolerance`, and the best evaluated point is returned: its
        cut is valid by construction, and its value is the c sent up.
        """
        start = time.time()
        solver = self.cgsp.cpm.solver
        x_hat = self._x_hat
        rho = self._rho

        # Q_s(x_hat), and y-bar for the first column (Algorithm 2, line 3)
        solver.activate_original_objective()
        solver.solve()
        if not solver.is_optimal():
            raise RuntimeError(
                f"Scenario {self.idx}: the recourse problem at the master's "
                "solution did not solve to optimality."
            )
        leaf_objective = solver.get_objective_value()
        solver.original_objective.deactivate()
        self._unfix_variables()

        # (x_hat, rho, y_bar) is in S^phi since rho >= phi(x_hat), and it
        # bounds the CGMP: alpha - beta^T x_hat - rho(1 + tau) <= q(y_bar) - rho
        first = self._column(x_hat, rho, leaf_objective, leaf_objective)
        self.cgmp.add_cuts([CutList([first])])
        status, solution, _ = self.cgmp.run_step(None)

        best: tuple[float, float, List[float], float] | None = None
        for _ in range(self.max_iteration):
            if solution is None or status == STATUS_INFEASIBLE:
                break
            tau, beta = solution[0], list(solution[1:])
            alpha = self.cgmp.cpm.get_theta_value(0)
            penalty = sum(b * x for b, x in zip(beta, x_hat)) + rho * (1 + tau)
            model_value = alpha - penalty

            self._update_cgsp_objective(beta, tau)
            self.cgsp.cpm.solve()
            if not solver.is_optimal():
                raise RuntimeError(
                    f"Scenario {self.idx}: the cut-generation subproblem did "
                    "not solve to optimality."
                )
            m = self.cgsp.get_objective_value()
            lower = m - penalty
            if best is None or lower > best[0]:
                best = (lower, m, beta, tau)
            if model_value - best[0] <= self.tolerance:
                break
            if status in (STATUS_MAX_ITERATION, STATUS_TIME_LIMIT):
                break

            column = self._column(
                [value(var) for var in self.cgsp.get_vars()],
                self.cgsp.cpm.get_theta_value(0),
                self.cgsp.get_original_objective_value(),
                m,
            )
            status, solution, _ = self.cgmp.run_step([CutList([column])])
        self.step_time.append(time.time() - start)

        # every column is a point of S^phi, valid for as long as the master's
        # cuts are unchanged (see _sync_cuts)
        self._cgmp_cuts = [c.cut for group in self.cgmp.cpm.get_cuts() for c in group]

        if best is None:
            raise RuntimeError(
                f"Scenario {self.idx}: the cut-generation master did not solve."
            )
        c, alpha, beta, tau = best
        cut = OptimalityCut(
            coeffs={i: val for i, val in enumerate(beta)},
            rhs=alpha,
            objective_value=leaf_objective,
            info={},
        )
        root_objective = self.cgsp.cpm.get_parent_objective_value()
        return BdScUpMessage(cut, c, tau, root_objective + leaf_objective)

    def _get_up_message_pbm(self) -> BdScUpMessage:
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

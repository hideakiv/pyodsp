from typing import List, Tuple
from pathlib import Path
import time
import pandas as pd
import logging

from pyomo.environ import ScalarVar, Var

from .message import (
    BdScInitDnMessage,
    BdScUpMessage,
    BdScDnMessage,
    BdScFinalDnMessage,
    BdScFinalUpMessage,
)
from ..node._alg import IAlgRoot, verify_sense_matches
from pyodsp.solver.pyomo_solver import PyomoSolver
from pyodsp.alg.bm.bm import BundleMethod
from pyodsp.alg.const import STATUS_NOT_FINISHED, STATUS_OPTIMAL, STATUS_STALLED
from pyodsp.alg.params import BM_ABS_TOLERANCE, BM_REL_TOLERANCE
from pyodsp.dec.node._message import NodeIdx
from pyodsp.dec.node.cut_aggregator import CutAggregator


def _verify_state_is_complete(solver: PyomoSolver) -> None:
    """Every first-stage variable must be a coupling variable.

    This algorithm was built for two-stage stochastic programs, and it
    prices columns over each subproblem's own feasible set — which
    contains a copy of the whole first stage (see examples/bdsc/cs.py).
    A first-stage variable the subproblems can set but never report is
    one the master never constrains them on, so the generated columns
    describe a larger set than the master is restricted to and the cuts
    built from them do not bound it. The run still converges; it
    converges on the wrong problem.

    The root's model is the first stage, so anything on it outside the
    coupling list is exactly that kind of variable.
    """
    coupled = {id(var) for var in solver.get_vars()}
    missing = []
    for component in solver.model.component_objects(Var, active=True):
        for index in component:
            var = component[index]
            if id(var) not in coupled:
                missing.append(var.name)

    if missing:
        raise ValueError(
            "Benders decomposition with scaled cuts needs every first-stage "
            f"variable in its coupling list, but {sorted(missing)[:5]}"
            f"{'...' if len(missing) > 5 else ''} "
            f"({len(missing)} in total) are not. Its subproblems each hold a "
            "copy of the first stage, so a variable they can set without "
            "reporting it leaves the master pricing against a feasible set "
            "wider than its own. Add them to the coupling list, or use plain "
            "Benders, whose master keeps the first stage to itself. Pass "
            "validate=False to skip this check."
        )


class BdScAlgRootBm(IAlgRoot):
    def __init__(
        self, solver: PyomoSolver, max_iteration=1000, validate: bool = True
    ) -> None:
        """
        Args:
            solver: The first-stage model, coupling on every one of its
                variables — see _verify_state_is_complete.
            max_iteration: Iteration cap for the master.
            validate: Whether to run the structural check above. Turning
                it off skips one pass over the model's variables; the
                requirement still holds, it just stops being enforced.
        """
        if not solver.is_minimize():
            raise ValueError(
                "Benders decomposition with scaled cuts needs a minimize model. PyomoSolver converts a "
                "maximize one on construction, so this solver was built "
                "with convert_maximize=False — that is reserved for the "
                "internal masters whose sense is deliberately inverted."
            )
        if validate:
            _verify_state_is_complete(solver)
        self.bm = BundleMethod(solver, max_iteration)
        self.step_time: List[float] = []

    def get_vars(self) -> List[ScalarVar]:
        return self.bm.get_vars()

    def build(
        self,
        groups: list[list[NodeIdx]],
        children_multipliers: dict[NodeIdx, float],
        children_bounds: dict[NodeIdx, float],
    ) -> None:
        num_cuts = len(groups)
        self.groups = groups
        self.children_multipliers = children_multipliers
        self.cut_aggregator = CutAggregator(self.groups, self.children_multipliers)
        self.subobj_bounds: List[float | None] = []
        for group in self.groups:
            bound = 0.0
            for member in group:
                if member not in children_bounds:
                    bound = None
                    break
                bound += self.children_multipliers[member] * children_bounds[member]
            self.subobj_bounds.append(bound)
        self.bm.build(num_cuts, self.subobj_bounds)
        self.rho = None
        self.solution = None
        self.objective = None
        # The best master solution evaluated so far and its true cost,
        # c^T x + sum_s p_s Q_s(x). The master's own upper bound is the
        # cost of its latest solution only, which a stalled run can leave
        # far above a solution it already passed through.
        self.best_objective: float | None = None
        self.best_solution: List[float] | None = None

    def run_step(
        self, up_messages: dict[NodeIdx, BdScUpMessage] | None
    ) -> Tuple[int, BdScDnMessage]:
        """Advance the master and tell the column-generation subproblems
        which cuts it now holds.

        The dn message carries the master's *whole* current cut set
        whenever the master takes a step, not just the cuts added by that
        step: a subproblem cannot arrive at the same set by replaying the
        additions, because the master also drops cuts (aging) and declines
        them (dominance, similarity) based on solves the subproblem never
        performs. When the master does not step, its cuts are unchanged and
        None is sent, meaning "keep what you have".
        """
        if up_messages is None:
            cuts_list = None
            start = time.time()
            status, solution, objective = self.bm.run_step(cuts_list)
            self.step_time.append(time.time() - start)
            self.rho = sum(self.bm.get_theta_value())
            self.solution = solution
            self.objective = objective
            dn_cut_list = self.bm.get_cut_list()
        else:
            c_average = 0.0
            tau_average = 0.0
            for node_idx, message in up_messages.items():
                c_average += message.get_c() * self.children_multipliers[node_idx]
                tau_average += message.get_tau() * self.children_multipliers[node_idx]
            if c_average <= 1e-6:
                cuts_list = self.cut_aggregator.get_aggregate_cuts(up_messages)
                for cuts in cuts_list:
                    for cut in cuts:
                        cut.rhs = cut.rhs / (1 + tau_average)
                        for var_id, coeff in cut.coeffs.items():
                            cut.coeffs[var_id] = coeff / (1 + tau_average)

                trial = self.solution
                assert trial is not None
                improvement = self._improvement_at(trial, cuts_list)

                start = time.time()
                status, solution, objective = self.bm.run_step(cuts_list)
                self.step_time.append(time.time() - start)
                # the cuts were computed at the previous solution, so the
                # incumbent run_step just recorded is that solution's cost
                self._record_incumbent(trial, self.bm.obj_val[-1])
                self.rho = sum(self.bm.get_theta_value())
                self.solution = solution
                self.objective = objective
                dn_cut_list = self.bm.get_cut_list()
                if status == STATUS_NOT_FINISHED:
                    status = self._termination_check(improvement)
            else:
                # the master did not step, so its cuts did not change
                dn_cut_list = None
                assert self.rho is not None
                self.rho = self.rho + c_average / (1 + tau_average)
                status = STATUS_NOT_FINISHED
        return status, BdScDnMessage(
            self.solution, self.rho, dn_cut_list, self.subobj_bounds, self.objective
        )

    def _improvement_at(self, trial: List[float], cuts_list) -> float:
        """How far the scaled cut lifts the outer approximation at the
        point it was computed for: its value there minus the master's
        theta there (the master has not been re-solved since)."""
        theta = sum(self.bm.get_theta_value())
        values = []
        for cuts in cuts_list:
            for cut in cuts:
                values.append(
                    cut.rhs - sum(coeff * trial[i] for i, coeff in cut.coeffs.items())
                )
        return max(values) - theta if values else 0.0

    def _record_incumbent(self, trial: List[float], value: float | None) -> None:
        if value is None:
            return
        if self.best_objective is None or value < self.best_objective:
            self.best_objective = value
            self.best_solution = list(trial)

    def _termination_check(self, improvement: float) -> int:
        """Stop on the gap to the best incumbent, or once the cuts stop
        improving (van der Laan & Romeijnders, Algorithm 1 and Section 2.2).

        The gap is the one BundleMethod checks, but against the best
        solution seen rather than the latest. The second test is the
        paper's practical stopping rule: a scaled cut need not be tight at
        the master's solution, and when it lifts the outer approximation
        there by less than the tolerance the master returns the same
        solution again, so the run would only repeat itself.
        """
        lower = self.bm.obj_bound[-1] if self.bm.obj_bound else None
        upper = self.best_objective
        if lower is not None and upper is not None:
            scale = max(abs(upper), BM_ABS_TOLERANCE)
            if (upper - lower) / scale < BM_REL_TOLERANCE:
                self.bm.status = STATUS_OPTIMAL
                self.bm.logger.log_status_optimal()
                return STATUS_OPTIMAL
        if improvement < BM_ABS_TOLERANCE:
            self.bm.status = STATUS_STALLED
            self.bm.logger.log_info(
                "Benders with scaled cuts stalled: the last cut improved the "
                f"outer approximation by {improvement:.3g} at the master's "
                "solution; returning the best incumbent"
            )
            return STATUS_STALLED
        return STATUS_NOT_FINISHED

    def add_cuts(self, up_messages: dict[NodeIdx, BdScUpMessage]) -> None:
        cuts_list = self.cut_aggregator.get_aggregate_cuts(up_messages)
        self.bm.add_cuts(cuts_list)

    def reset_iteration(self) -> None:
        self.bm.reset_iteration()

    def get_final_dn_message(self, **kwargs) -> BdScFinalDnMessage:
        # Hand out the best solution seen, not the master's latest, and
        # leave it on the master's variables: the reported first stage and
        # its cost are read from there.
        if self.best_solution is not None:
            for var, val in zip(self.get_vars(), self.best_solution):
                var.set_value(val)
        return BdScFinalDnMessage([var.value for var in self.get_vars()])

    def pass_final_up_message(
        self, messages: dict[NodeIdx, BdScFinalUpMessage]
    ) -> BdScFinalUpMessage:
        obj = self.bm.get_original_objective_value()
        for message in messages.values():
            child_obj = message.get_objective()
            if child_obj is None:
                return BdScFinalUpMessage(None)
            obj += child_obj
        return BdScFinalUpMessage(obj)

    def get_num_vars(self) -> int:
        return len(self.get_vars())

    def get_init_dn_message(self, **kwargs) -> BdScInitDnMessage:
        return BdScInitDnMessage()

    def save(self, dir: Path) -> None:
        self.bm.save(dir)
        path = dir / "step_time.csv"
        df = pd.DataFrame(self.step_time, columns=["step_time"])
        df.to_csv(path, index=False)

    def set_sense_multiplier(self, multiplier: float) -> None:
        verify_sense_matches(self.get_sense_multiplier(), multiplier)

    def get_sense_multiplier(self) -> float:
        return self.bm.get_solver().sense_multiplier

    def is_minimize(self) -> bool:
        # Always: PyomoSolver converts a maximize model on construction.
        return True

    def set_logger(self, node_id: int, depth: int, level: int = logging.INFO) -> None:
        self.bm.set_logger(node_id, depth, level)

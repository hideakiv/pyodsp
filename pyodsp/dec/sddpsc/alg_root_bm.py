from typing import List, Tuple
from pathlib import Path
import time
import pandas as pd
import logging

from pyomo.environ import ScalarVar

from ..bdsc.message import (
    BdScInitDnMessage,
    BdScUpMessage,
    BdScDnMessage,
    BdScFinalDnMessage,
    BdScFinalUpMessage,
)
from ..node._alg import IAlgRoot, verify_sense_matches
from pyodsp.solver.pyomo_solver import PyomoSolver
from pyodsp.alg.bm.bm import BundleMethod
from pyodsp.dec.node._message import NodeIdx
from pyodsp.dec.node.cut_aggregator import CutAggregator

# A reply's tau at or below this counts as zero: its cut no longer depends
# on the origin's theta (see SddpScAlgRootBm._role_for).
TAU_ZERO = 1e-9


class SddpScAlgRootBm(IAlgRoot):
    """A stage's root in an SDDP lattice, building its cost-to-go cuts with
    bdsc's scaled cuts instead of plain Benders duals.

    Reuses bdsc.BdScAlgLeafPyomo unchanged as each stage's leaf -- the
    column-generation machinery (CGSP/CGMP, row generation) doesn't need to
    know it's embedded in a multistage lattice rather than a two-stage
    master. What differs is this class, and only in how the rho fixed point
    (van der Laan & Romeijnders, Lemma 2-4) is driven: the two-stage master
    loops run_step with the same trial point until rho converges, but
    Lattice visits a node once per outer SDDP iteration and has no hook for
    an inner loop -- run_step(None) only steps the bundle method against
    whatever cuts it already has (the forward pass), and add_cuts, called
    separately on the backward pass, both advances rho and adds a cut if
    the children's current rho has closed the gap enough. rho refines
    across SDDP's own many outer iterations instead of inside one visit.

    This is deliberate, not merely a concession to Lattice's interface:
    under a deep (the default: maximum) lookback, a leaf's CGSP grows with
    the stage index (see pyodsp.dec.sddpsc's module docstring), so
    re-running row generation many times per visit -- once per rho
    iterate -- would multiply an already-growing per-visit cost. One rho
    step per visit keeps each visit's cost to one row-generation pass.

    Lattice calls add_cuts on every node at a stage every iteration,
    whether or not that node was the one actually solved by that
    iteration's forward pass (siblings share one children set, so a node
    can go many iterations without being resampled). rho is therefore
    seeded once in build(), from the same floor the leaves themselves
    start theta at, rather than on the first run_step -- a node's first
    add_cuts can otherwise arrive before it has ever been stepped.
    """

    def __init__(
        self,
        solver: PyomoSolver,
        max_iteration: int = 1000,
        name_prefix: str = "",
        state_params: List | None = None,
        purgeable: bool = True,
    ) -> None:
        """
        Args:
            solver: This stage's own model, coupling on the state it hands
                to the next stage -- see bdsc.alg_root_bm for the
                first-stage-completeness requirement this inherits; it
                applies the same way here, to this stage's own variables.
            max_iteration: Iteration cap for this stage's bundle method.
            name_prefix: Forwarded to this stage's BundleMethod. Only
                matters if solver.model is ever deliberately shared with
                another cut-adding master; in general an inner node's root
                role and leaf role use separate models (see state_params),
                so this is rarely needed in practice.
            state_params: This role's own mutable pyo.Param copies of the
                state its leaf role (if any) receives from the parent, in
                the same order as the leaf role's own coupling_up list.
                SddpScNodeInner.pass_dn_message pushes the trial point into
                these on every visit (see set_state), so the root role's
                solve sees the same incoming state the leaf role was just
                fixed to, despite the two roles using separate models. None
                (the default) for a node with no incoming state, i.e. the
                true first-stage root.
            purgeable: False on an MPI replica, as for BdAlgRootBm: a
                replica mirrors rank 0's cuts (apply_snapshot) and must
                not age or decline any on its own -- see LatticeMpi.
        """
        if not solver.is_minimize():
            raise ValueError(
                "SDDP with scaled cuts needs a minimize model. PyomoSolver "
                "converts a maximize one on construction, so this solver "
                "was built with convert_maximize=False -- that is reserved "
                "for the internal masters whose sense is deliberately "
                "inverted."
            )
        self.bm = BundleMethod(
            solver,
            max_iteration,
            force=not purgeable,
            purgeable=purgeable,
            name_prefix=name_prefix,
        )
        self.state_params = list(state_params) if state_params is not None else []
        self.step_time: List[float] = []
        self.rho: float | None = None
        self.solution: List[float] | None = None
        self.objective: float | None = None
        self.idx: NodeIdx | None = None
        # Whether add_cuts actually added a cut since the last run_step --
        # only then has the trial point potentially moved, so only then
        # does rho's baseline need resyncing (see run_step).
        self._cuts_changed = False

    def get_vars(self) -> List[ScalarVar]:
        return self.bm.get_vars()

    def set_state(self, values: List[float]) -> None:
        """Update this role's own copy of the incoming state -- see
        state_params. Called by SddpScNodeInner, not directly by users.
        """
        if len(values) != len(self.state_params):
            raise ValueError(
                f"Got {len(values)} state value(s), but this node has "
                f"{len(self.state_params)} state_params; they must match "
                "the leaf role's own coupling_up list one for one."
            )
        for param, val in zip(self.state_params, values):
            param.set_value(val)

    def build(
        self,
        groups: list[list[NodeIdx]],
        children_multipliers: dict[NodeIdx, float],
        children_bounds: dict[NodeIdx, float],
    ) -> None:
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
        self.bm.build(len(self.groups), self.subobj_bounds)
        # rho_0 = phi(x_bar), the outer approximation's value at the trial
        # point (Lemma 4) -- before any real cuts exist, that's exactly the
        # floor the leaves themselves install under theta from these same
        # bounds (see BdScAlgLeafPyomo._set_subobj_bound).
        self.rho = sum(self.subobj_bounds)
        self._cuts_changed = False

    def run_step(
        self, up_messages: dict[NodeIdx, BdScUpMessage] | None
    ) -> Tuple[int, BdScDnMessage]:
        """Step the bundle method against the cuts it already has.

        Lattice's forward pass always calls this with up_messages=None --
        new cuts arrive separately, through add_cuts on the backward pass.
        The up_messages branch is kept so this is still a complete
        IAlgRoot on its own, the way BdAlgRootBm's is, but Lattice never
        exercises it.
        """
        cuts_list = None
        if up_messages is not None:
            cuts_list = self.cut_aggregator.get_aggregate_cuts(up_messages)
        start = time.time()
        status, solution, objective = self.bm.run_step(cuts_list)
        self.step_time.append(time.time() - start)
        self.solution = solution
        self.objective = objective
        # Resync rho's baseline (Lemma 4: rho_0 = phi(x_bar)) only when the
        # trial point may actually have moved -- i.e. a cut was added since
        # the last time this stepped. Resyncing unconditionally would
        # overwrite every bit of progress add_cuts made refining rho toward
        # rho* on visits where the gap wasn't closed enough to add a cut
        # yet, since the bundle method's theta hasn't moved either.
        if cuts_list is not None or self._cuts_changed:
            self.rho = sum(self.bm.get_theta_value())
            self._cuts_changed = False
        message = BdScDnMessage(
            self.solution,
            self.rho,
            self.bm.get_cut_list(),
            self.subobj_bounds,
            self.objective,
        )
        # so the replies priced against this rho and these cuts can be told
        # apart from a sibling's -- see add_cuts
        message.set_origin(self.idx, dict(self.children_multipliers))
        return status, message

    def add_cuts(self, up_messages: dict[NodeIdx, BdScUpMessage]) -> None:
        """One step of the rho fixed point (Lemma 2-4): advance rho toward
        rho*, and only add a cut once the children's reported gap c has
        closed enough that it is worth keeping. No re-solve here, matching
        BdAlgRootBm.add_cuts -- this stage's next run_step picks up
        whatever this adds.
        """
        assert self.rho is not None, "add_cuts called before build()"

        role = self._role_for(up_messages)
        if role is None:
            return

        # Whether the cut is worth keeping is the origin's call: c measures
        # the gap at its rho, so weigh the replies with its probabilities.
        # A "shared" master (a Markov sibling) then reweights the cut with
        # its own, which _role_for has checked is valid (every tau == 0).
        if role == "origin":
            gate = self.children_multipliers
        else:
            gate = self._origin_multipliers(up_messages)
        c_average = 0.0
        for node_idx, message in up_messages.items():
            if node_idx in gate:
                c_average += message.get_c() * gate[node_idx]
        tau_average = 0.0
        for node_idx, message in up_messages.items():
            if node_idx in self.children_multipliers:
                tau_average += message.get_tau() * self.children_multipliers[node_idx]

        if c_average <= 1e-6:
            cuts_list = self.cut_aggregator.get_aggregate_cuts(up_messages)
            for cuts in cuts_list:
                for cut in cuts:
                    cut.rhs = cut.rhs / (1 + tau_average)
                    for var_id, coeff in cut.coeffs.items():
                        cut.coeffs[var_id] = coeff / (1 + tau_average)
            self.bm.add_cuts(cuts_list)
            self._cuts_changed = True
        elif role == "origin":
            self.rho = self.rho + c_average / (1 + tau_average)

    def _own_replies(self, up_messages):
        return [
            message
            for node_idx, message in up_messages.items()
            if node_idx in self.children_multipliers
        ]

    def _origin_multipliers(self, up_messages) -> dict:
        return next(
            m.get_origin_multipliers() for m in self._own_replies(up_messages)
        )

    def _role_for(self, up_messages: dict[NodeIdx, BdScUpMessage]) -> str | None:
        """How this master may use these replies: "origin", "sibling",
        "shared" or None (not at all).

        Lattice hands every master at a stage the same replies, but they
        were priced against one master's trial point: the rho and cuts in
        the dn message the children last received. Only that master (the
        origin) may take the rho step -- c measures the gap at *its* rho
        and x_hat, and means nothing for anyone else's.

        The cut is another matter. Each child's scaled cut is
        alpha <= q(y) + beta^T x + tau * theta over S^phi, with theta the
        origin's cost-to-go, and the origin combines them as
        (1 + sum_s p_s tau_s) theta >= sum_s p_s (alpha_s - beta_s^T x).
        Who else may take it:

        - "sibling": same children at the same probabilities (a
          stagewise-independent stage). Its aggregate is the origin's.
        - "shared": different probabilities (a Markov lattice), but every
          child reported tau == 0. Then theta drops out of each child's
          cut, which becomes an ordinary cut Q_s(x) >= alpha_s - beta_s^T x
          over S^phi's projection onto x -- the stage's own feasible set,
          the same for every node of the stage since sddpsc requires a
          shared transition. Valid on its own, so any probabilities may
          reweight it.
        - None otherwise: for a Markov sibling, a cut with tau > 0 is tied
          to the origin's theta, and so to the origin's expectation. That
          master leaves these replies alone and learns when it is visited.

        Either way the cut is only taken when the origin would take it
        (its c, at its probabilities, is closed); see add_cuts.

        Replies that carry no origin (bdsc's own leaf, which does not echo
        it) are taken as this master's own, which is all a single-master
        stage ever sees.
        """
        own = self._own_replies(up_messages)
        origins = {message.get_origin() for message in own}
        if origins <= {None, self.idx}:
            return "origin"
        if len(origins) != 1:
            return None
        multipliers = own[0].get_origin_multipliers()
        if multipliers is None:
            return None
        if set(multipliers) == set(self.children_multipliers) and all(
            abs(multipliers[child] - p) <= 1e-12
            for child, p in self.children_multipliers.items()
        ):
            return "sibling"
        # a shared cut needs a reply from every child of this master, and
        # the origin's gate needs one from every child of the origin
        if not set(self.children_multipliers) <= set(up_messages):
            return None
        if not set(multipliers) <= set(up_messages):
            return None
        if all(
            abs(up_messages[child].get_tau()) <= TAU_ZERO
            for child in self.children_multipliers
        ):
            return "shared"
        return None

    def get_snapshot(self) -> dict:
        """Everything a replica needs to step exactly as this master would
        next: its cuts, and the rho fixed point's progress.

        Plain SDDP's replicas need only the cuts (see LatticeMpi). Here the
        dn message also carries rho, which a replica's own add_cuts never
        advances, and whether the next run_step resyncs it -- which the
        backward pass decides, and only rank 0 runs that.
        """
        return {
            "cuts": self.bm.get_cut_list(),
            "rho": self.rho,
            "cuts_changed": self._cuts_changed,
        }

    def apply_snapshot(self, snapshot: dict) -> None:
        """Become a copy of the master get_snapshot was taken from."""
        self.bm.replace_cuts(snapshot["cuts"])
        self.rho = snapshot["rho"]
        self._cuts_changed = snapshot["cuts_changed"]

    def reset_iteration(self) -> None:
        self.bm.reset_iteration()

    def get_final_dn_message(self, **kwargs) -> BdScFinalDnMessage:
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

    def restore_cuts(self, dir: Path) -> None:
        """Reinstate the value function saved by a previous run's save."""
        self.bm.restore_cuts(dir)

    def set_sense_multiplier(self, multiplier: float) -> None:
        verify_sense_matches(self.get_sense_multiplier(), multiplier)

    def get_sense_multiplier(self) -> float:
        return self.bm.get_solver().sense_multiplier

    def get_solver(self) -> PyomoSolver:
        return self.bm.get_solver()

    def is_minimize(self) -> bool:
        # Always: PyomoSolver converts a maximize model on construction.
        return True

    def set_logger(self, node_id: int, depth: int, level: int = logging.INFO) -> None:
        # this is where a master learns its own node's index (as
        # BdScAlgLeafPyomo does) -- needed to tag its dn messages
        self.idx = node_id
        self.bm.set_logger(node_id, depth, level)

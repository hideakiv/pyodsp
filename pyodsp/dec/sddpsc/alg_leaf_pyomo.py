from typing import List

from pyomo.environ import (
    ConstraintList,
    Objective,
    RangeSet,
    Reals,
    ScalarVar,
    Var,
    minimize,
    value,
)

from pyodsp.alg.bm.cuts import CutList, FeasibilityCut, OptimalityCut
from pyodsp.alg.params import BM_ABS_TOLERANCE
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig

from ..bdsc.alg_leaf_pyomo import BdScAlgLeafPyomo

__all__ = ["SddpScAlgLeafPyomo"]

# A stage's leaf is bdsc's two-stage leaf: its column-generation subproblem
# (CGSP) already embeds a free copy of whatever its model declares outside
# its own coupling variables (see bdsc's _verify_state_is_complete and
# examples/bdsc/cs.py, where the leaf embeds the root's first-stage model
# wholesale). For a stage-t leaf here, that model is the deterministic
# transition chain from stage t-K through t-1 (K the lookback, see
# pyodsp.dec.sddpsc's module docstring), built as ordinary free,
# non-coupling variables -- only x_{t-1} is declared as the coupling
# variable, which is what keeps the resulting scaled cut a function of
# x_{t-1} alone despite the richer embedded feasible region.
#
# What it adds is the one thing a two-stage leaf never needs: at an inner
# stage, the recourse it prices is not just this stage's cost but this
# stage's cost plus the next stage's cost-to-go -- see the future term below.


class SddpScAlgLeafPyomo(BdScAlgLeafPyomo):
    """bdsc's leaf, plus the next stage's cost-to-go for an inner stage.

    An inner node's leaf role reports a cut on x_{t-1} for the cost of
    stage t *and everything after it*. Its model only states stage t's
    cost, so the rest comes from the node's own root role: the cuts that
    role holds on its outgoing state x_t approximate Q_{t+1}(x_t), and this
    leaf mirrors them onto its own copy of x_t (`future_vars`) under a
    theta of its own. The recourse it prices is then

        q(y) = (stage t's cost) + sum_g theta_future[g]

    in the Q_s(x_hat) solve, the cgsp, and each column's q. Since the
    mirrored cuts only ever under-approximate Q_{t+1}, a scaled cut built
    against an older copy stays valid -- just weaker, as in plain SDDP.

    What a sample path pays at this stage is still the stage's own cost:
    the up message's objective leaves theta_future out, so simulated costs
    never count a cost-to-go estimate as incurred.

    Without future_vars, or until build_future is called, this behaves
    exactly like BdScAlgLeafPyomo -- which is what a last-stage leaf wants.
    """

    def __init__(
        self,
        solver: PyomoSolver,
        master_config: SolverConfig,
        max_iteration=1000,
        cut_master: str = "bm",
        tolerance: float = BM_ABS_TOLERANCE,
        coef_bound: float | None = None,
        name_prefix: str = "",
        future_vars: List[ScalarVar] | None = None,
    ):
        """
        Args:
            future_vars: This leaf model's own copy of the stage's outgoing
                state x_t, in the same order as the node's root role's
                coupling_dn list -- the mirrored cuts index into it by
                position. None for a last-stage leaf, which has no future.
            Others: see BdScAlgLeafPyomo.
        """
        super().__init__(
            solver,
            master_config,
            max_iteration,
            cut_master=cut_master,
            tolerance=tolerance,
            coef_bound=coef_bound,
            name_prefix=name_prefix,
        )
        self.future_vars = list(future_vars) if future_vars is not None else None
        self._future_prefix = f"{name_prefix}_future"
        self._future_theta = None
        self._future_obj = None
        self._future_keys: set | None = None

    def _future_name(self, suffix: str) -> str:
        return f"{self._future_prefix}{suffix}"

    def has_future(self) -> bool:
        return self._future_theta is not None

    def build_future(self, subobj_bounds: List[float | None]) -> None:
        """Add theta_future, one per cut group of the node's root role.

        Called once the root role is built, since that is where the groups
        and their bounds are settled (SddpScAlgRootBm.subobj_bounds): each
        theta_future[g] gets the same floor the root role's own theta[g]
        has, which keeps the cgsp bounded before any cut arrives.
        """
        if self.future_vars is None:
            raise ValueError(
                "This leaf was built without future_vars, so it has no copy "
                "of the stage's outgoing state to hang a cost-to-go on."
            )
        if self.has_future():
            raise ValueError("build_future was already called on this leaf")
        if any(bound is None for bound in subobj_bounds):
            raise ValueError(
                "SDDP with scaled cuts needs a bound on every child's "
                "objective to floor the next stage's cost-to-go; call "
                "set_bound on each child node"
            )

        model = self.cgsp.cpm.solver.model
        bounds = list(subobj_bounds)
        model.add_component(
            self._future_name("_theta"),
            Var(
                RangeSet(0, len(bounds) - 1),
                domain=Reals,
                bounds=lambda _, g: (bounds[g], None),
            ),
        )
        self._future_theta = getattr(model, self._future_name("_theta"))
        self._add_cut_list()
        self._future_keys = set()

        # The recourse objective, kept beside the original one rather than
        # replacing it: the original still has to report the stage's own
        # cost on its own (see _get_up_message_row_generation).
        obj = Objective(expr=self._recourse_expr(), sense=minimize)
        model.add_component(self._future_name("_obj"), obj)
        obj.deactivate()
        self._future_obj = obj

    def _add_cut_list(self) -> None:
        model = self.cgsp.cpm.solver.model
        name = self._future_name("_cuts")
        if model.component(name) is not None:
            model.del_component(name)
        model.add_component(name, ConstraintList())

    def set_future_cuts(self, cut_list: List[CutList]) -> None:
        """Mirror the root role's whole cut set (bm.get_cut_list()).

        Replaces wholesale, like _sync_cuts does for the parent's cuts. Any
        change also retires the cgmp's collected columns: a column records
        q at the point it was generated, and q includes theta_future, so a
        cut added *or* dropped here changes what that column is worth.
        """
        if not self.has_future():
            raise ValueError("set_future_cuts called before build_future")
        if len(cut_list) != len(self._future_theta):
            raise ValueError(
                f"Got {len(cut_list)} cut group(s), but this leaf's future "
                f"term has {len(self._future_theta)}; they must match the "
                "node's root role's cut groups one for one."
            )
        incoming = self._cut_keys(cut for group in cut_list for cut in group)
        if incoming == self._future_keys:
            return
        self._future_keys = incoming
        self._cgmp_cuts = []
        if getattr(self, "cgmp", None) is not None:
            # pass_dn_message already built this trial point's cgmp and
            # seeded it with the columns just retired -- and in a lattice
            # the root role's cuts change after that (backward pass), before
            # get_up_message runs. Rebuild it from the same trial point.
            self._create_master(self._x_hat, self._rho)

        self._add_cut_list()
        constraints = self.cgsp.cpm.solver.model.component(self._future_name("_cuts"))
        x = self.future_vars
        for g, group in enumerate(cut_list):
            for cut in group:
                lhs = sum(coeff * x[j] for j, coeff in cut.coeffs.items())
                if isinstance(cut, OptimalityCut):
                    # the form CuttingPlaneMethod._add_optimality_cut uses,
                    # for a minimize master (which the root role always is)
                    constraints.add(lhs + self._future_theta[g] >= cut.rhs)
                elif isinstance(cut, FeasibilityCut):
                    constraints.add(lhs >= cut.rhs)
                else:
                    raise TypeError(f"Unknown cut type {type(cut).__name__}")

    def pass_dn_message(self, message) -> None:
        # whose trial point this is, echoed on the reply -- see
        # SddpScAlgRootBm.add_cuts
        self._origin = (message.get_origin(), message.get_origin_multipliers())
        super().pass_dn_message(message)

    def get_up_message(self):
        message = super().get_up_message()
        message.set_origin(*getattr(self, "_origin", (None, None)))
        return message

    def _future_expr(self):
        return sum(self._future_theta[g] for g in self._future_theta)

    def _recourse_expr(self):
        base = super()._recourse_expr()
        if not self.has_future():
            return base
        return base + self._future_expr()

    def _activate_recourse_objective(self) -> None:
        if not self.has_future():
            super()._activate_recourse_objective()
            return
        model = self.cgsp.cpm.solver.model
        for obj in model.component_data_objects(Objective, active=True):
            obj.deactivate()
        self._future_obj.activate()

    def _deactivate_recourse_objective(self) -> None:
        if not self.has_future():
            super()._deactivate_recourse_objective()
            return
        self._future_obj.deactivate()

    def _recourse_value(self) -> float | None:
        stage = super()._recourse_value()
        if not self.has_future() or stage is None:
            return stage
        return stage + value(self._future_expr())

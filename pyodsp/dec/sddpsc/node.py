from ..node.dec_node import DecNodeInner, DecNodeParent, DecNodeChild
from ..bdsc.message import BdScDnMessage, BdScFinalDnMessage, BdScUpMessage
from .alg_leaf_pyomo import SddpScAlgLeafPyomo


class SddpScNodeInner(DecNodeInner):
    """An inner stage's node, whose root role and leaf role use separate
    pyomo models (see sddpsc's module docstring: sharing one model between
    two cut-adding bundle methods -- the leaf's own cgsp and the root's own
    bm -- needs more care than plain Benders' inner nodes do, since both
    sides add private components, including an Objective, to it).

    With separate models, nothing carries anything between the two roles
    for free, so this node does it, in both directions:

    - down: after the usual pass_dn_message, the trial point goes into the
      root role's own mutable Param copies of the incoming state
      (SddpScAlgRootBm.state_params / set_state);
    - up: the root role's cuts on the outgoing state -- this stage's
      approximation of the next stage's cost-to-go -- are mirrored into the
      leaf role's future term (SddpScAlgLeafPyomo.set_future_cuts), so the
      cut the leaf role reports covers everything downstream, not just this
      stage's own cost.

    The mirror is refreshed whenever the leaf role is about to solve, not
    only after add_cuts: the root role's bundle method also ages cuts out
    during its own run_step, and set_future_cuts is a no-op when nothing
    changed.
    """

    def __init__(self, idx, alg_root, alg_leaf, **kwargs) -> None:
        if not isinstance(alg_leaf, SddpScAlgLeafPyomo) or alg_leaf.future_vars is None:
            raise ValueError(
                "An inner sddpsc node's leaf role must be an SddpScAlgLeafPyomo "
                "built with future_vars (its own copy of the stage's outgoing "
                "state); without them the cut it reports ignores every stage "
                "after this one."
            )
        super().__init__(idx, alg_root, alg_leaf, **kwargs)

    def build_inner(self) -> None:
        # The root role first: its build settles the cut groups and their
        # bounds, which the leaf role's future term mirrors.
        DecNodeParent.build_inner(self)
        DecNodeChild.build_inner(self)
        self.alg_leaf.build_future(self.alg_root.subobj_bounds)
        self._sync_future()

    def _sync_future(self) -> None:
        self.alg_leaf.set_future_cuts(self.alg_root.bm.get_cut_list())

    def pass_dn_message(self, message: BdScDnMessage) -> None:
        self._sync_future()
        super().pass_dn_message(message)
        self.alg_root.set_state(message.get_solution())

    def add_cuts(self, up_messages) -> None:
        super().add_cuts(up_messages)
        self._sync_future()

    def get_up_message(self) -> BdScUpMessage:
        self._sync_future()
        return super().get_up_message()

    def pass_final_dn_message(self, message: BdScFinalDnMessage) -> None:
        self._sync_future()
        return super().pass_final_dn_message(message)

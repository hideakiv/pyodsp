from ..node.dec_node import DecNodeInner
from ..bdsc.message import BdScDnMessage


class SddpScNodeInner(DecNodeInner):
    """An inner stage's node, whose root role and leaf role use separate
    pyomo models (see sddpsc's module docstring: sharing one model between
    two cut-adding bundle methods -- the leaf's own cgsp and the root's own
    bm -- needs more care than plain Benders' inner nodes do, since both
    sides add private components, including an Objective, to it).

    With separate models, nothing else propagates the incoming state from
    the leaf role (which pass_dn_message fixes it on) to the root role's
    own solve. This node does that one thing: after the usual
    pass_dn_message, it pushes the same trial point into the root role's
    own mutable Param copies (SddpScAlgRootBm.state_params / set_state).
    """

    def pass_dn_message(self, message: BdScDnMessage) -> None:
        super().pass_dn_message(message)
        self.alg_root.set_state(message.get_solution())

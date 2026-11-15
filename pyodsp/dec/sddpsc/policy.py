"""The decision rule an sddpsc run leaves behind.

Plain SDDP solves a stage by fixing the coupling of the model its inner
node's two roles share (see pyodsp.dec.sddp.policy). sddpsc gives the
roles separate models (see pyodsp.dec.sddpsc), so a stage is solved where
its decision and cost-to-go actually live:

    root and inner  the root role's own model -- the stage alone, against
                    the cuts it learned. An inner node takes the incoming
                    state through its state_params (SddpScAlgRootBm.
                    set_state), as on every visit during the run.
    last stage      the leaf role's transition chain, with the incoming
                    state (the chain's coupling variables) fixed and the
                    stage's own cost as the objective -- the same solve
                    BdScAlgLeafPyomo.pass_final_dn_message does. The
                    chain's earlier stages only have to be feasible for
                    that state, which a state reached along the same
                    transitions always is.
"""

from typing import List

from pyomo.environ import Block

from ..node._node import INode, INodeParent
from ..sddp.policy import SddpPolicy, StageSolution

__all__ = ["SddpScPolicy"]


class SddpScPolicy(SddpPolicy):
    """SddpPolicy for nodes built for sddpsc -- see the module docstring."""

    def _state_size(self, node: INode) -> int:
        if isinstance(node, INodeParent):
            return len(node.alg_root.state_params)
        return len(node.alg_leaf.cgsp.cpm.solver.vars)

    def _stage_solver(self, node: INode, prev_state: List[float] | None):
        if isinstance(node, INodeParent):
            if prev_state is not None:
                node.alg_root.set_state(list(prev_state))
            return node.alg_root.get_solver()
        leaf = node.alg_leaf
        leaf._fix_variables(list(prev_state))
        leaf._activate_recourse_objective()
        return leaf.cgsp.cpm.solver

    def _release(self, node: INode) -> None:
        if isinstance(node, INodeParent):
            return
        # leave the leaf as pass_final_dn_message does, ready for the next
        # dn message to set it up for column generation
        node.alg_leaf._deactivate_recourse_objective()
        node.alg_leaf._unfix_variables()

    def _read_solution(self, node: INode, solver) -> StageSolution:
        solution = super()._read_solution(node, solver)
        if not isinstance(node, INodeParent):
            solution.solution = _own_stage_values(solver)
        return solution


def _own_stage_values(solver) -> dict:
    """The last stage's own variables, named as in the stage builder.

    A leaf chain holds stages t - lookback .. t as Blocks in order (see
    pyodsp.model.msp.builders_sddpsc); the earlier ones are only there to
    shape the feasible region, so their values are no decision of this
    stage's. A leaf model without Blocks (built by hand) is reported whole.
    """
    values = solver.get_variable_values()
    blocks = list(solver.model.component_objects(Block, descend_into=False))
    if not blocks:
        return values
    prefix = f"{blocks[-1].name}."
    return {
        name[len(prefix):]: value
        for name, value in values.items()
        if name.startswith(prefix)
    }

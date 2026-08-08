from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import logging

from ..bd.message import BdDnMessage
from ..graph.lattice import Lattice
from ..node._message import NodeIdx
from ..node._node import INode, INodeParent, INodeChild
from .logger import SddpLogger


@dataclass
class StageSolution:
    """What one stage decides, given the state handed to it."""

    node_idx: NodeIdx
    stage_cost: float
    """This stage's own cost, excluding the approximated cost-to-go."""
    future_cost: float
    """The cut approximation's price on the state this stage passes on."""
    total_cost: float
    """stage_cost + future_cost — what this stage actually minimized."""
    next_state: List[float]
    """Values of the coupling variables handed to the next stage (empty at
    the last stage, which has none)."""
    solution: Dict[str, float]
    """Every variable's value at this solve, keyed by name."""


class SddpPolicy:
    """The decision rule an SDDP run leaves behind.

    A converged run's real output is not the last forward pass's solution
    — that one is conditioned on whichever scenario happened to be sampled
    last. It is the value function each stage learned, stored as cuts.
    Given that, a stage's decision for *any* incoming state is well
    defined: fix the coupling variables to that state and re-solve the
    stage against its cuts. This class does exactly that.

    Use it either on the nodes still in memory after a run, or on rebuilt
    nodes via from_saved, which reads back the cuts that Lattice._save
    wrote. Note that evaluating leaves a node's coupling variables fixed
    at the state just evaluated, exactly as a forward pass would — every
    call re-fixes them, so results do not depend on call order.
    """

    def __init__(self, nodes: List[List[INode]]) -> None:
        self.nodes: Dict[NodeIdx, INode] = {
            node.get_idx(): node for stage in nodes for node in stage
        }

    def evaluate(
        self, node_idx: NodeIdx, prev_state: List[float] | None = None
    ) -> StageSolution:
        """Solve one stage for the given previous-stage state.

        Pass prev_state=None for the first stage, which has no incoming
        state.
        """
        node = self._get_node(node_idx)
        takes_state = isinstance(node, INodeChild)

        if prev_state is None:
            if takes_state:
                raise ValueError(
                    f"Node {node_idx} couples to a previous stage; "
                    "prev_state is required"
                )
        else:
            if not takes_state:
                raise ValueError(
                    f"Node {node_idx} has no previous stage to take a state from"
                )
            expected = len(node.alg_leaf.get_solver().get_vars())
            if len(prev_state) != expected:
                raise ValueError(
                    f"Node {node_idx} couples on {expected} variable(s), "
                    f"got a state of length {len(prev_state)}"
                )
            # the objective passed along only affects reporting, not the solve
            node.pass_dn_message(BdDnMessage(list(prev_state), 0.0))

        alg = node.alg_leaf if takes_state else node.alg_root
        solver = alg.get_solver()
        solver.solve()
        if not solver.is_optimal():
            raise ValueError(
                f"Node {node_idx} did not solve to optimality for the given state"
            )

        stage_cost = solver.get_original_objective_value()
        # a stage that has children optimizes stage cost + theta; one that
        # does not has no theta, so the two coincide
        total_cost = solver.get_objective_value()
        if isinstance(node, INodeParent):
            next_state = [var.value for var in node.alg_root.get_vars()]
        else:
            next_state = []

        return StageSolution(
            node_idx=node_idx,
            stage_cost=stage_cost,
            future_cost=total_cost - stage_cost,
            total_cost=total_cost,
            next_state=next_state,
            solution=solver.get_variable_values(),
        )

    def simulate(self, path: List[NodeIdx]) -> List[StageSolution]:
        """Walk a scenario path stage by stage, feeding each stage's state
        to the next.

        `path` names one node per stage, starting at the first stage. The
        realized cost of the trajectory is the sum of the stage_costs (not
        of the total_costs, which double-count the future).
        """
        if not path:
            raise ValueError("path is empty")
        solutions: List[StageSolution] = []
        state: List[float] | None = None
        previous: INode | None = None
        for position, node_idx in enumerate(path):
            if previous is not None and node_idx not in previous.get_children():
                raise ValueError(
                    f"Node {node_idx} is not a child of node {previous.get_idx()}"
                )
            solution = self.evaluate(node_idx, state)
            solutions.append(solution)
            if not solution.next_state and position < len(path) - 1:
                raise ValueError(
                    f"Node {node_idx} is a last-stage node and passes on no state, "
                    f"but the path continues to {path[position + 1]}"
                )
            state = solution.next_state
            previous = self._get_node(node_idx)
        return solutions

    def restore(self, filedir: Path) -> None:
        """Load each stage's saved cuts back into its (already built) master."""
        for node_idx, node in self.nodes.items():
            if isinstance(node, INodeParent):
                node.alg_root.restore_cuts(Path(filedir) / f"node{node_idx}")

    @classmethod
    def from_saved(
        cls,
        nodes: List[List[INode]],
        filedir: Path,
        level: int = logging.WARNING,
    ) -> "SddpPolicy":
        """Rebuild a policy from a completed run's output directory.

        `nodes` must be constructed the same way the run constructed them
        — same models, same children, same groups. Only the learned cuts
        come from disk; the models themselves are yours to rebuild.
        """
        lattice = Lattice(nodes, SddpLogger(level), Path(filedir))
        # the same build/bounds handshake a run does before its first
        # iteration; without it the masters have no theta to attach cuts to
        lattice._run_init()
        policy = cls(nodes)
        policy.restore(filedir)
        return policy

    def _get_node(self, node_idx: NodeIdx) -> INode:
        if node_idx not in self.nodes:
            raise ValueError(f"No node with idx {node_idx}")
        return self.nodes[node_idx]

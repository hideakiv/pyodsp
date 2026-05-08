from typing import List, Dict
from pathlib import Path

import numpy as np
import scipy.stats as st
from pyodsp.alg.bm.cuts import OptimalityCut, FeasibilityCut

from ..node._logger import ILogger
from ..node._node import INode, INodeRoot, INodeLeaf, INodeInner
from ..node._message import (
    DnMessage,
    NodeIdx,
)
from ..utils import create_directory


from pyodsp.alg.params import SDDP_REL_TOLERANCE, SDDP_IMPROVE_TOLERANCE


class Lattice:
    def __init__(
        self,
        nodes: List[List[INode]],
        logger: ILogger,
        filedir: Path,
        max_iteration: int = 1000,
        sample_frequency: int = 10,
        sample_size: int = 1000,
        confidence_level: float = 0.95,
    ) -> None:
        self.num_stages = len(nodes)
        self._verify_nodes(nodes)
        self.logger = logger
        self.filedir = filedir
        self.max_iteration = max_iteration
        self.sample_frequency = sample_frequency
        self.sample_size = sample_size
        self.confidence_level = confidence_level
        self.is_minimize = True
        create_directory(self.filedir)

        self.prev_samples = None

    def _verify_nodes(self, nodes: List[List[INode]]) -> None:
        self.root: INodeRoot | None = None
        self.leaves: List[INodeLeaf] = []
        self.nodes: Dict[NodeIdx, INode] = {}
        self.stages: Dict[int, List[NodeIdx]] = {k: [] for k in range(self.num_stages)}
        for stage in range(self.num_stages):
            for node in nodes[stage]:
                self.nodes[node.get_idx()] = node
                self.stages[stage].append(node.get_idx())
            if stage == 0:
                if len(nodes[stage]) > 1:
                    raise ValueError(
                        f"Number of nodes is {len(nodes[stage])} in stage {stage}."
                    )
                node = nodes[stage][0]
                if isinstance(node, INodeLeaf):
                    raise ValueError(f"Stage {stage} must be root node.")
                assert isinstance(node, INodeRoot)
                assert not isinstance(node, INodeLeaf)
                self.root = node
            elif stage == self.num_stages - 1:
                for node in nodes[stage]:
                    if isinstance(node, INodeRoot):
                        raise ValueError(f"Stage {stage} must be leaf node.")
                    assert isinstance(node, INodeLeaf)
                    assert not isinstance(node, INodeRoot)
                    self.leaves.append(node)
            else:
                for node in nodes[stage]:
                    if not isinstance(node, INodeInner):
                        raise ValueError(f"Stage {stage} must be inner node.")

    def run(self, init_solution: DnMessage | None = None):
        self.logger.log_initialization()
        self._run_init()
        self._run_main()
        self._save()

    def _run_init(self) -> None:
        if self.root is None:
            raise ValueError("Root node not found")
        self.root.set_depth(0)
        for stage in range(self.num_stages - 1):
            self._run_init_forward(stage)

        for stage in range(self.num_stages - 1, 0, -1):
            self._run_init_backward(stage)

    def _run_init_forward(self, stage: int) -> None:
        assert stage < self.num_stages - 1
        for node_idx in self.stages[stage]:
            node = self.nodes[node_idx]
            assert isinstance(node, INodeRoot)
            node.set_logger()

        for node_idx in self.stages[stage]:
            node = self.nodes[node_idx]
            assert isinstance(node, INodeRoot)
            init_dn_message = node.get_init_dn_message()

            if stage == 0:
                self.is_minimize = init_dn_message.get_is_minimize()

            for child_idx in self.stages[stage + 1]:
                child = self.nodes[child_idx]
                assert isinstance(child, INodeLeaf)
                child.pass_init_dn_message(init_dn_message)

    def _run_init_backward(self, stage: int) -> None:
        assert stage > 0
        init_up_messages = {}
        for child_id in self.stages[stage]:
            child = self.nodes[child_id]
            assert isinstance(child, INodeLeaf)
            child.build()
            init_up_message = child.get_init_up_message()
            init_up_messages[child_id] = init_up_message

        for node_idx in self.stages[stage - 1]:
            node = self.nodes[node_idx]
            assert isinstance(node, INodeRoot)
            node.pass_init_up_messages(init_up_messages)
            node.build()
            node.reset()

    def _run_main(self) -> None:
        if self.root is None:
            raise ValueError("Root node not found")
        bound = -1e9
        for iteration in range(self.max_iteration):
            np.random.seed(42 + self.sample_size + iteration)
            bound = self._run_root()
            if iteration % self.sample_frequency == self.sample_frequency - 1:
                if self._termination(bound):
                    break
            else:
                self._run_forwards()

            bound = self._run_backwards()

    def _termination(self, bound: float) -> bool:
        objectives = []
        for _ in range(self.sample_size):
            np.random.seed(42 + _)
            objective = self._run_forwards()
            objectives.append(objective)
        ci_d, ci_u = st.t.interval(
            confidence=self.confidence_level,
            df=len(objectives) - 1,
            loc=np.mean(objectives),
            scale=st.sem(objectives),
        )
        self.logger.log_info(
            f"lower: {ci_d}, upper: {ci_u}, confidence: {self.confidence_level}"
        )
        converged = False
        if self.is_minimize:
            converged = (
                abs(ci_u - bound) / max(abs(ci_u), abs(bound)) < SDDP_REL_TOLERANCE
            )
        else:
            converged = (
                abs(ci_d - bound) / max(abs(ci_d), abs(bound)) < SDDP_REL_TOLERANCE
            )

        if converged:
            self.logger.log_info("SDDP termination with convergence.")
            return True

        no_improve = False
        if self.prev_samples is not None:
            sample_diffs = [
                self.prev_samples[i] - objectives[i] for i in range(len(objectives))
            ]
            all_zero = True
            for sample_diff in sample_diffs:
                if sample_diff > 1e-9:
                    all_zero = False
                    break

            if all_zero:
                no_improve = True
            else:
                diff_ci_d, diff_ci_u = st.t.interval(
                    confidence=self.confidence_level,
                    df=len(sample_diffs) - 1,
                    loc=np.mean(sample_diffs),
                    scale=st.sem(sample_diffs),
                )
                no_improve = diff_ci_u < SDDP_IMPROVE_TOLERANCE

        if no_improve:
            self.logger.log_info("SDDP termination with no improvement.")
            return True

        self.prev_samples = objectives

        return False

    def _run_root(self) -> float:
        assert self.root is not None
        self._run_forward(self.root)

        return self.root.alg_root.bm.get_objective_value()  # FIXME: properly access

    def _run_forwards(self) -> float:
        node = self.root
        assert node is not None
        path = [node.get_idx()]
        for stage in range(1, self.num_stages):
            # randomly sample node in the next stage
            prob = [node.get_multiplier(node_idx) for node_idx in node.get_children()]
            sampled_idx = np.random.choice(node.get_children(), p=prob)
            node = self.nodes[sampled_idx]
            path.append(node.get_idx())

            if stage < self.num_stages - 1:
                assert isinstance(node, INodeRoot)
                self._run_forward(node)
        assert not isinstance(node, INodeRoot)
        assert isinstance(node, INodeLeaf)
        up_message = node.get_up_message()  # solve leaf node
        return up_message.get_objective()

    def _run_forward(self, node: INodeRoot) -> float:
        node.reset()
        status, dn_message = node.run_step(None)

        for child_id in node.get_children():
            child = self.nodes[child_id]
            assert isinstance(child, INodeLeaf)
            child.pass_dn_message(dn_message)
        return dn_message.get_objective()

    def _run_backwards(self) -> None:
        for stage in range(self.num_stages - 1, 0, -1):
            self._run_backward(stage)

    def _run_backward(self, stage: int) -> None:
        assert stage > 0
        up_messages = {}
        for child_id in self.stages[stage]:
            child = self.nodes[child_id]
            assert isinstance(child, INodeLeaf)
            up_message = child.get_up_message()
            cut_dn = up_message.get_cut()
            assert cut_dn is not None
            if isinstance(cut_dn, OptimalityCut):
                self.logger.log_sub_problem(
                    child.get_idx(), "Optimality", cut_dn.coeffs, cut_dn.rhs
                )
            if isinstance(cut_dn, FeasibilityCut):
                self.logger.log_sub_problem(
                    child.get_idx(), "Feasibility", cut_dn.coeffs, cut_dn.rhs
                )
            up_messages[child_id] = up_message

        for node_idx in self.stages[stage - 1]:
            node = self.nodes[node_idx]
            assert isinstance(node, INodeRoot)
            node.add_cuts(up_messages)

    def _save(self) -> None:
        for node in self.nodes.values():
            node.save(self.filedir)

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


from pyodsp.alg.params import BM_REL_TOLERANCE


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

            # check parents
            if stage > 0:
                previous = self.stages[stage - 1]
                for node in nodes[stage]:
                    parents = node.get_parents()
                    if set(previous) != set(parents):
                        raise ValueError(
                            f"parents f{parents} are not the same as previous {previous}"
                        )

    def run(self, init_solution: DnMessage | None = None):
        self.logger.log_initialization()
        self._run_init()
        self._run_main()
        self.logger.log_finaliziation()
        final_obj = self._run_final()
        self.logger.log_completion(final_obj)
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
        node = self.nodes[
            self.stages[stage][0]
        ]  # get first node in stage as representative
        assert isinstance(node, INodeRoot)
        init_dn_message = node.get_init_dn_message()

        if stage == 0:
            self.is_minimize = init_dn_message.get_is_minimize()

        for node_idx in self.stages[stage + 1]:
            child = self.nodes[node_idx]
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
            objective = self._run_forwards()
            objectives.append(objective)
        ci_d, ci_u = st.t.interval(
            confidence=self.confidence_level,
            df=len(objectives) - 1,
            loc=np.mean(objectives),
            scale=st.sem(objectives),
        )
        if self.is_minimize:
            print(abs(ci_u - bound) / max(abs(ci_u), abs(bound)))
            breakpoint()
            return abs(ci_u - bound) / max(abs(ci_u), abs(bound)) < BM_REL_TOLERANCE
        else:
            return abs(ci_d - bound) / max(abs(ci_d), abs(bound)) < BM_REL_TOLERANCE

    def _run_root(self) -> float:
        assert self.root is not None
        self._run_forward(self.root)

        return (
            self.root.alg_root.bm.cpm.solver.get_objective_value()
        )  # FIXME: properly access

    def _run_forwards(self) -> float:
        node = self.root
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
        return up_message.get_objective()  # FIXME: reference only in BdUpMessage

    def _run_forward(self, node: INodeRoot) -> float:
        node.reset()
        status, dn_message = node.run_step(None)

        for child_id in node.get_children():
            child = self.nodes[child_id]
            assert isinstance(child, INodeLeaf)
            child.pass_dn_message(dn_message)
        return dn_message.get_objective()  # FIXME: reference only in BdDnMessage

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

    def _run_final(self) -> float | None:
        if self.root is None:
            raise ValueError("root node not found")
        for stage in range(self.num_stages - 1):
            self._run_final_forward(stage)

        values: List[float | None] = []
        for stage in range(self.num_stages, 0, -1):
            values = self._run_final_backward(stage)

        return values[0]

    def _run_final_forward(self, stage: int) -> None:
        assert stage < self.num_stages - 1
        node = self.stages[stage][0]  # get first node in stage as representative
        assert isinstance(node, INodeRoot)
        final_dn_message = node.get_final_dn_message()

        for node_idx in self.stages[stage + 1]:
            child = self.nodes[node_idx]
            assert isinstance(child, INodeLeaf)
            child.pass_final_dn_message(final_dn_message)

    def _run_final_backward(self, stage: int) -> List[float | None]:
        assert stage > 0
        final_up_messages = {}
        for child_id in self.stages[stage]:
            child = self.nodes[child_id]
            assert isinstance(child, INodeLeaf)
            final_up_message = child.get_final_up_message()
            final_up_messages[child_id] = final_up_message

        values: List[float | None] = []
        for node_idx in self.stages[stage - 1]:
            node = self.nodes[node_idx]
            assert isinstance(node, INodeRoot)
            value = node.pass_final_up_message(final_up_messages)
            values.append(value.get_objective())

        return values

    def _save(self) -> None:
        for node in self.nodes.values():
            node.save(self.filedir)

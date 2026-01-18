from typing import List, Dict
from pathlib import Path

from pyodsp.alg.bm.cuts import OptimalityCut, FeasibilityCut

from ..node._logger import ILogger
from ..node._node import INode, INodeRoot, INodeLeaf, INodeInner
from ..node._message import (
    InitDnMessage,
    InitUpMessage,
    DnMessage,
    UpMessage,
    FinalDnMessage,
    FinalUpMessage,
    NodeIdx,
)
from ..utils import create_directory

from pyodsp.alg.const import (
    STATUS_NOT_FINISHED,
    STATUS_MAX_ITERATION,
    STATUS_TIME_LIMIT,
)


class Lattice:
    def __init__(
        self, nodes: List[List[INode]], logger: ILogger, filedir: Path
    ) -> None:
        self.num_stages = len(nodes)
        self._verify_nodes(nodes)
        self.logger = logger
        self.filedir = filedir
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
                assert type(node) is INodeRoot
                self.root = node
            elif stage == self.num_stages - 1:
                for node in nodes[stage]:
                    if isinstance(node, INodeRoot):
                        raise ValueError(f"Stage {stage} must be leaf node.")
                    assert type(node) is INodeLeaf
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

        for stage in range(self.num_stages, 0, -1):
            self._run_init_backward(stage)

    def _run_init_forward(self, stage: int) -> None:
        assert stage < self.num_stages - 1
        for node_idx in self.stages[stage]:
            node = self.nodes[node_idx]
            assert isinstance(node, INodeRoot)
            node.set_logger()
        node = self.stages[stage][0]  # get first node in stage as representative
        assert isinstance(node, INodeRoot)
        init_dn_message = node.get_init_dn_message()

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
        while True:
            for stage in range(self.num_stages - 1):
                self._run_forward(stage)

            for stage in range(self.num_stages, 0, -1):
                self._run_backward(stage)

    def _run_forward(self, stage: int) -> None:
        assert stage < self.num_stages - 1
        node = self.stages[stage][0]  # TODO: implement node selection
        assert isinstance(node, INodeRoot)
        node.reset()
        status, dn_message = node.run_step(None)

        for node_idx in self.stages[stage + 1]:
            child = self.nodes[node_idx]
            assert isinstance(child, INodeLeaf)
            child.pass_dn_message(dn_message)

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

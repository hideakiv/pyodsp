from typing import List, Dict
from pathlib import Path

from pyodsp.alg.bm.cuts import OptimalityCut, FeasibilityCut

from ..node._logger import ILogger
from ..node._node import INode, INodeRoot, INodeLeaf, INodeInner
from ..node._message import (
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


class Tree:
    def __init__(self, nodes: List[INode], logger: ILogger, filedir: Path) -> None:
        self._verify_nodes(nodes)
        self.logger = logger
        self.filedir = filedir
        create_directory(self.filedir)

    def _verify_nodes(self, nodes: List[INode]) -> None:
        self.root: INodeRoot | None = None
        self.leaves: List[INodeLeaf] = []
        self.nodes: Dict[NodeIdx, INode] = {}
        for node in nodes:
            self.nodes[node.get_idx()] = node
            if isinstance(node, INodeInner):
                continue
            if isinstance(node, INodeRoot):
                if self.root is not None:
                    raise ValueError("Multiple root nodes found")
                self.root = node
                continue
            if isinstance(node, INodeLeaf):
                self.leaves.append(node)
                continue

            raise ValueError(f"Unknown object of type {type(node)} detected")

    def run(self, init_solution: DnMessage | None = None):
        self.logger.log_initialization()
        self._run_init()
        up_messages = self._run_main_preprocess(init_solution)
        self._run_main(up_messages)
        self.logger.log_finaliziation()
        final_obj = self._run_final()
        self.logger.log_completion(final_obj)
        self._save()

    def get_num_root_vars(self) -> int:
        if self.root is None:
            raise ValueError("root node not found")
        return self.root.get_num_vars()

    def _run_init(self) -> None:
        if self.root is None:
            raise ValueError("Root node not found")
        self.root.set_depth(0)
        self._run_init_core(self.root)

    def _run_init_core(self, node: INode) -> InitUpMessage | None:
        if isinstance(node, INodeRoot):
            node.set_logger()
            init_up_messages = {}
            for child_id in node.get_children():
                init_dn_message = node.get_init_dn_message(child_id=child_id)
                child = self.nodes[child_id]
                assert isinstance(child, INodeLeaf)
                child.pass_init_dn_message(init_dn_message)
                init_up_messages[child_id] = self._run_init_core(child)
            node.pass_init_up_messages(init_up_messages)
            node.build()
            node.reset()
        if isinstance(node, INodeLeaf):
            node.build()
            return node.get_init_up_message()
        return None

    def _run_main_preprocess(
        self, init_solution: DnMessage | None
    ) -> Dict[NodeIdx, UpMessage] | None:
        if self.root is None:
            raise ValueError("root node not found")
        if init_solution is None:
            up_messages = None
        else:
            up_messages = {}
            for child_id in self.root.get_children():
                child = self.nodes[child_id]
                up_messages[child_id] = self._run_node(child, init_solution)
        return up_messages

    def _run_main(self, up_messages: Dict[NodeIdx, UpMessage] | None) -> None:
        if self.root is None:
            raise ValueError("Root node not found")
        self._run_node_core(self.root, up_messages)

    def _run_node(
        self, node: INode, dn_message: DnMessage | None = None
    ) -> UpMessage | None:
        if isinstance(node, INodeRoot):
            if isinstance(node, INodeInner):
                assert dn_message is not None
                node.pass_dn_message(dn_message)

            return self._run_node_core(node, None)
        elif isinstance(node, INodeLeaf):
            assert dn_message is not None
            return node.solve(dn_message)
        else:
            raise ValueError(f"Unknown object of type {type(node)} detected")

    def _run_node_core(
        self, node: INodeRoot, up_messages: Dict[NodeIdx, UpMessage] | None
    ) -> UpMessage | None:
        node.reset()
        up_messages = None
        while True:
            status, new_dn_message = node.run_step(up_messages)

            if status != STATUS_NOT_FINISHED:
                if isinstance(node, INodeInner):
                    if status == STATUS_MAX_ITERATION or status == STATUS_TIME_LIMIT:
                        up_messages = self._get_up_messages(node, new_dn_message)
                        node.add_cuts(up_messages)
                    return node.get_up_message()
                else:
                    return

            up_messages = self._get_up_messages(node, new_dn_message)

    def _get_up_messages(
        self, node: INode, dn_message: DnMessage
    ) -> Dict[NodeIdx, UpMessage]:
        up_messages = {}
        for child_id in node.get_children():
            child = self.nodes[child_id]
            assert isinstance(child, INodeLeaf)
            up_message = self._get_up_message(child, dn_message)
            up_messages[child_id] = up_message
        return up_messages

    def _get_up_message(self, node: INodeLeaf, message: DnMessage) -> UpMessage:
        up_message = self._run_node(node, message)
        assert up_message is not None
        cut_dn = up_message.get_cut()
        assert cut_dn is not None
        if isinstance(cut_dn, OptimalityCut):
            self.logger.log_sub_problem(
                node.get_idx(), "Optimality", cut_dn.coeffs, cut_dn.rhs
            )
        if isinstance(cut_dn, FeasibilityCut):
            self.logger.log_sub_problem(
                node.get_idx(), "Feasibility", cut_dn.coeffs, cut_dn.rhs
            )
        return up_message

    def _run_final(self) -> float:
        if self.root is None:
            raise ValueError("root node not found")

        message = self._run_final_core(self.root)
        return message.get_objective()

    def _run_final_core(
        self, node: INode, dn_message: FinalDnMessage | None = None
    ) -> FinalUpMessage:
        if isinstance(node, INodeLeaf):
            assert dn_message is not None
            node.pass_final_dn_message(dn_message)
        if isinstance(node, INodeRoot):
            up_messages = {}
            for child_id in node.get_children():
                child = self.nodes[child_id]
                new_dn_message = node.get_final_dn_message(
                    node_id=child_id, groups=node.get_groups()
                )
                up_messages[child_id] = self._run_final_core(child, new_dn_message)
            return node.pass_final_up_message(up_messages)
        elif isinstance(node, INodeLeaf):
            return node.get_final_up_message()
        else:
            raise ValueError(f"Unknown object of type {type(node)} detected")

    def _save(self) -> None:
        for node in self.nodes.values():
            node.save(self.filedir)

from typing import List, Dict
from pathlib import Path

from pyodsp.alg.cuts import OptimalityCut, FeasibilityCut
from pyodsp.alg.const import *

from .logger import BdLogger
from ..utils import create_directory
from ..node._node import INode, INodeRoot, INodeLeaf, INodeInner
from ..run._message import DnMessage, UpMessage, NodeIdx


class BdRun:
    def __init__(self, nodes: List[INode], filedir: Path):
        self.nodes: Dict[int, INode] = {node.get_idx(): node for node in nodes}
        self.root = self._get_root()
        self.logger = BdLogger()

        self.filedir = filedir
        create_directory(self.filedir)

    def _get_root(self) -> INodeRoot | None:
        for node in self.nodes.values():
            if isinstance(node, INodeRoot) and not isinstance(node, INodeLeaf):
                return node
        return None

    def run(self):
        if self.root is not None:
            self.logger.log_initialization()
            self.root.set_depth(0)
            self._run_init(self.root)
            self._run_node(self.root)
            self._run_finalize(self.root)
        for node in self.nodes.values():
            node.save(self.filedir)

    def _run_init(self, node: INode) -> None:
        if isinstance(node, INodeRoot):
            node.set_logger()
            init_message = node.get_init_message()

        for child_id in node.get_children():
            child = self.nodes[child_id]
            assert isinstance(child, INodeLeaf)
            child.pass_init_message(init_message)
            self._run_init(child)

    def _run_node(
        self, node: INode, dn_message: DnMessage | None = None
    ) -> UpMessage | None:
        if isinstance(node, INodeRoot):
            self._set_bounds(node)
            node.build()
            if isinstance(node, INodeInner):
                assert dn_message is not None
                node.pass_dn_message(dn_message)

            node.reset()
            up_messages = None
            while True:
                status, new_dn_message = node.run_step(up_messages)

                if status != STATUS_NOT_FINISHED:
                    if isinstance(node, INodeInner):
                        if (
                            status == STATUS_MAX_ITERATION
                            or status == STATUS_TIME_LIMIT
                        ):
                            up_messages = self._get_up_messages(node, new_dn_message)
                            node.add_cuts(up_messages)
                        return node.get_up_message()
                    else:
                        return

                up_messages = self._get_up_messages(node, new_dn_message)
        if isinstance(node, INodeLeaf):
            assert dn_message is not None
            node.build()
            return node.solve(dn_message)

    def _run_finalize(self, node: INode) -> None:
        if isinstance(node, INodeRoot):
            final_message = node.get_final_message()
            for child_id in node.get_children():
                child = self.nodes[child_id]
                assert isinstance(child, INodeLeaf)
                child.pass_final_message(final_message)
                self._run_finalize(child)

    def _get_up_messages(
        self, node: INode, dn_message: DnMessage
    ) -> Dict[NodeIdx, UpMessage]:
        up_messages = {}
        for child in node.get_children():
            up_message = self._get_up_message(child, dn_message)
            up_messages[child] = up_message
        return up_messages

    def _get_up_message(self, idx: int, dn_message: DnMessage) -> UpMessage:
        up_message = self._run_node(self.nodes[idx], dn_message)
        assert up_message is not None
        cut_dn = up_message.get_cut()
        if isinstance(cut_dn, OptimalityCut):
            self.logger.log_sub_problem(idx, "Optimality", cut_dn.coeffs, cut_dn.rhs)
        if isinstance(cut_dn, FeasibilityCut):
            self.logger.log_sub_problem(idx, "Feasibility", cut_dn.coeffs, cut_dn.rhs)
        return up_message

    def _set_bounds(self, node: INodeRoot) -> None:
        for child in node.get_children():
            child_node = self.nodes[child]
            assert isinstance(child_node, INodeLeaf)
            node.set_child_bound(child, child_node.get_bound())

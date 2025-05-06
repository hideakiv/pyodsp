from typing import List, Dict
from pathlib import Path

from pyodsp.alg.cuts import OptimalityCut, FeasibilityCut
from pyodsp.alg.const import *

from .logger import DdLogger
from .message import DdDnMessage
from .mip_heuristic_root import MipHeuristicRoot
from ..utils import create_directory
from ..node._node import INode, INodeRoot, INodeLeaf
from ..node._message import InitMessage, FinalMessage, DnMessage, UpMessage, NodeIdx


class DdRun:
    def __init__(self, nodes: List[INode], filedir: Path):
        self.nodes: Dict[int, INode] = {node.get_idx(): node for node in nodes}
        self.root = self._get_root()
        self.logger = DdLogger()

        self.filedir = filedir
        create_directory(self.filedir)

    def _get_root(self) -> INodeRoot | None:
        for node in self.nodes.values():
            if isinstance(node, INodeRoot) and not isinstance(node, INodeLeaf):
                return node
        return None

    def run(self, init_solution: List[float] | None = None):
        if self.root is not None:
            # run root process
            self.root.set_depth(0)
            self.root.set_logger()
            self._init_root()
            for child_id in self.root.get_children():
                self._init_leaf(
                    child_id,
                    self.root.get_init_message(child_id=child_id),
                )

            self._run_root(init_solution)
            self._finalize_root()
        else:
            raise ValueError("root node not found")

        for node in self.nodes.values():
            node.save(self.filedir)

    def _init_root(self) -> None:
        self.logger.log_initialization()
        assert self.root is not None
        self.root.build()

    def _init_leaf(
        self,
        node_id: int,
        message: InitMessage,
    ) -> None:
        node = self.nodes[node_id]
        assert isinstance(node, INodeLeaf)
        node.pass_init_message(message)

    def _run_root(self, init_solution: List[float] | None = None) -> None:
        assert self.root is not None
        self.root.reset()
        if init_solution is None:
            dn_message = DdDnMessage([0.0 for _ in range(self.root.get_num_vars())])
        else:
            dn_message = DdDnMessage(init_solution)
        up_messages = self._run_leaf(dn_message)
        while True:
            status, dn_message = self.root.run_step(up_messages)
            if status != STATUS_NOT_FINISHED:
                break
            up_messages = self._run_leaf(dn_message)

        self.logger.log_finaliziation()

    def _run_leaf(self, message: DnMessage) -> Dict[NodeIdx, UpMessage]:
        up_messages = {}
        for node in self.nodes.values():
            if isinstance(node, INodeLeaf):
                up_message = self._get_up_message(node.get_idx(), message)
                up_messages[node.get_idx()] = up_message
        return up_messages

    def _get_up_message(self, idx: int, message: DnMessage) -> UpMessage:
        node = self.nodes[idx]
        assert isinstance(node, INodeLeaf)
        node.build()
        up_message = node.solve(message)
        cut_dn = up_message.get_cut()
        assert cut_dn is not None
        if isinstance(cut_dn, OptimalityCut):
            self.logger.log_sub_problem(idx, "Optimality", cut_dn.coeffs, cut_dn.rhs)
        if isinstance(cut_dn, FeasibilityCut):
            self.logger.log_sub_problem(idx, "Feasibility", cut_dn.coeffs, cut_dn.rhs)
        return up_message

    def _finalize_root(self) -> None:
        assert self.root is not None
        final_obj = 0.0
        for node_id in self.root.get_children():
            message = self.root.get_final_message(
                node_id=node_id, groups=self.root.get_groups()
            )
            sub_obj = self._finalize_leaf(node_id, message)
            final_obj += sub_obj
        self.logger.log_completion(final_obj)

    def _finalize_leaf(self, node_id, message: FinalMessage) -> float:
        node = self.nodes[node_id]
        assert isinstance(node, INodeLeaf)
        node.pass_final_message(message)
        return node.get_objective_value()

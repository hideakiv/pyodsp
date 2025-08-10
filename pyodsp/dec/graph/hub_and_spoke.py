from typing import List, Dict, Tuple
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

from pyodsp.alg.const import STATUS_NOT_FINISHED


class HubAndSpoke:
    def __init__(self, nodes: List[INode], logger: ILogger, filedir: Path) -> None:
        self._verify_nodes(nodes)
        self.logger = logger
        self.filedir = filedir
        create_directory(self.filedir)

    def _verify_nodes(self, nodes: List[INode]) -> None:
        self.root: INodeRoot | None = None
        self.leaves: List[INodeLeaf] = []
        for node in nodes:
            if isinstance(node, INodeInner):
                raise ValueError(f"Use of {type(node)} prohibited in HubAndSpoke")
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
        self._run_init_dn()
        self._run_init_up()
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

    def _run_init_dn(self) -> None:
        if self.root is None:
            raise ValueError("root node not found")
        self._init_root()
        for leaf in self.leaves:
            init_message = self.root.get_init_dn_message(child_id=leaf.get_idx())
            self._init_leaf(leaf, init_message)

    def _init_root(self) -> None:
        if self.root is None:
            raise ValueError("Root node not found")
        self.root.set_depth(0)
        self.root.set_logger()

    def _run_init_up(self) -> None:
        if self.root is None:
            raise ValueError("root node not found")
        messages = self._run_init_up_core()
        self.root.pass_init_up_messages(messages)
        self.root.build()

    def _run_init_up_core(self) -> Dict[NodeIdx, InitUpMessage]:
        messages = {}
        for leaf in self.leaves:
            init_message = leaf.get_init_up_message()
            messages[leaf.get_idx()] = init_message
        return messages

    def _init_leaf(self, node: INodeLeaf, message: InitDnMessage) -> None:
        node.pass_init_dn_message(message)

    def _run_main_preprocess(
        self, init_solution: DnMessage | None
    ) -> Dict[NodeIdx, UpMessage] | None:
        if self.root is not None:
            self.root.reset()
        if init_solution is None:
            up_messages = None
        else:
            up_messages = self._run_leaf(init_solution)
        return up_messages

    def _run_main(self, up_messages: Dict[NodeIdx, UpMessage] | None) -> None:
        while True:
            status, dn_message = self._run_root(up_messages)
            if status != STATUS_NOT_FINISHED:
                break
            up_messages = self._run_leaf(dn_message)

    def _run_root(
        self, up_messages: Dict[NodeIdx, UpMessage] | None
    ) -> Tuple[int, DnMessage]:
        assert self.root is not None
        return self.root.run_step(up_messages)

    def _run_leaf(self, message: DnMessage) -> Dict[NodeIdx, UpMessage]:
        up_messages = {}
        for node in self.leaves:
            up_message = self._get_up_message(node, message)
            up_messages[node.get_idx()] = up_message
        return up_messages

    def _get_up_message(self, node: INodeLeaf, message: DnMessage) -> UpMessage:
        node.build()
        up_message = node.solve(message)
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

    def _run_final(self) -> float | None:
        if self.root is None:
            raise ValueError("root node not found")

        up_messages = self._run_final_core()
        final_message = self.root.pass_final_up_message(up_messages)

        return final_message.get_objective()

    def _run_final_core(self) -> Dict[NodeIdx, FinalUpMessage]:
        if self.root is None:
            raise ValueError("root node not found")
        self._finalize_root()
        up_messages = {}
        for leaf in self.leaves:
            dn_message = self.root.get_final_dn_message(
                node_id=leaf.get_idx(), groups=self.root.get_groups()
            )
            up_message = self._finalize_leaf(leaf, dn_message)
            up_messages[leaf.get_idx()] = up_message
        return up_messages

    def _finalize_root(self) -> None:
        return

    def _finalize_leaf(
        self, node: INodeLeaf, final_message: FinalDnMessage
    ) -> FinalUpMessage:
        node.pass_final_dn_message(final_message)
        return node.get_final_up_message()

    def _save(self) -> None:
        self._save_root()
        for node in self.leaves:
            self._save_leaf(node)

    def _save_root(self) -> None:
        if self.root is None:
            raise ValueError("root node not found")
        self.root.save(self.filedir)

    def _save_leaf(self, node: INodeLeaf) -> None:
        node.save(self.filedir)

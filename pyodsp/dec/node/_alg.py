from abc import ABC, abstractmethod
from typing import List, Tuple
from pathlib import Path

from pyodsp.alg.bm.cuts import CutList

from ._message import (
    NodeIdx,
    InitDnMessage,
    InitUpMessage,
    FinalDnMessage,
    FinalUpMessage,
    DnMessage,
    UpMessage,
)


class IAlg(ABC):
    @abstractmethod
    def set_logger(self, idx: NodeIdx, depth: int, level: int) -> None:
        pass

    @abstractmethod
    def save(self, dir: Path) -> None:
        pass

    @abstractmethod
    def is_minimize(self) -> bool:
        pass

    @abstractmethod
    def get_sense_multiplier(self) -> float:
        """-1.0 if this node's model was written as a maximize problem.

        PyomoSolver converts such a model on construction, so the
        algorithms only ever see minimize problems. This is what the node
        reports with, and what tells a parent whether its children were
        written in the same sense it was.
        """


def verify_sense_matches(own: float, reported: float) -> None:
    """Raise unless a child reports the sense this node was written in.

    PyomoSolver converts a maximize model on construction, which erases
    the difference the algorithms would otherwise trip over — so the
    check compares the multipliers each solver recorded when it
    converted, which is a memory of what the user actually wrote.
    """
    if own != reported:
        raise ValueError(
            "Inconsistent optimization sense: this node's model is a "
            f"{'maximize' if own < 0 else 'minimize'} problem but a child's "
            "is not. Every node of one decomposition must be written in "
            "the same sense."
        )


class IAlgRoot(IAlg, ABC):
    @abstractmethod
    def set_sense_multiplier(self, multiplier: float) -> None:
        """Told the sense a child's model was written in.

        A root whose own model came from the user checks the two agree
        (see verify_sense_matches). Dual decomposition's root has no user
        model of its own — its master is synthesized from the coupling
        constraints — so it adopts the sense instead, and reconciles the
        children against each other.
        """

    @abstractmethod
    def build(
        self,
        groups: list[list[NodeIdx]],
        children_multipliers: dict[NodeIdx, float],
        children_bounds: dict[NodeIdx, float],
    ) -> None:
        pass

    @abstractmethod
    def run_step(
        self, up_messages: dict[NodeIdx, UpMessage] | None
    ) -> Tuple[int, DnMessage]:
        pass

    @abstractmethod
    def get_init_dn_message(self, **kwargs) -> InitDnMessage:
        pass

    @abstractmethod
    def add_cuts(self, up_messages: dict[NodeIdx, UpMessage]) -> None:
        pass

    @abstractmethod
    def reset_iteration(self) -> None:
        pass

    @abstractmethod
    def get_final_dn_message(self, **kwargs) -> FinalDnMessage:
        pass

    @abstractmethod
    def pass_final_up_message(
        self, messages: dict[NodeIdx, FinalUpMessage]
    ) -> FinalUpMessage:
        pass

    @abstractmethod
    def get_num_vars(self) -> int:
        pass


class IAlgLeaf(IAlg, ABC):
    @abstractmethod
    def build(self) -> None:
        pass

    @abstractmethod
    def pass_init_dn_message(self, message: InitDnMessage) -> None:
        pass

    @abstractmethod
    def get_init_up_message(self) -> InitUpMessage:
        pass

    @abstractmethod
    def pass_dn_message(self, message: DnMessage) -> None:
        pass

    @abstractmethod
    def pass_final_dn_message(self, message: FinalDnMessage) -> None:
        pass

    @abstractmethod
    def get_final_up_message(self) -> FinalUpMessage:
        pass

    @abstractmethod
    def get_up_message(self) -> UpMessage:
        pass

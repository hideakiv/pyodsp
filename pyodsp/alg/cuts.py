from typing import List, Dict, TypeVar, Generic
from dataclasses import dataclass


@dataclass
class Cut:
    """Base class for cuts."""

    coeffs: List[float]
    rhs: float

    info: Dict


@dataclass
class OptimalityCut(Cut):
    """Class for optimality cuts."""

    objective_value: float


@dataclass
class FeasibilityCut(Cut):
    """Class for feasibility cuts."""

    pass


T = TypeVar("T", bound=Cut)


class CutList(list[T], Generic[T]):
    def __init__(self, items: list[T] | None = None):
        super().__init__(items or [])

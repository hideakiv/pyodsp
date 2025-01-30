from typing import List
from dataclasses import dataclass


@dataclass
class Cut:
    """Base class for cuts."""

    coeffs: List[float]
    rhs: float


@dataclass
class OptimalityCut(Cut):
    """Class for optimality cuts."""

    objective_value: float


@dataclass
class FeasibilityCut(Cut):
    """Class for feasibility cuts."""

    pass


@dataclass
class WrappedCut:
    """Class for wrapping cuts with info"""

    cut: Cut
    iteration: int
    trial_point: List[float]
    period: int

from typing import List
from dataclasses import dataclass


@dataclass
class Cut:
    """Base class for cuts."""

    coefficients: List[float]
    constant: float


@dataclass
class OptimalityCut(Cut):
    """Class for optimality cuts."""

    objective_value: float


@dataclass
class FeasibilityCut(Cut):
    """Class for feasibility cuts."""

    pass

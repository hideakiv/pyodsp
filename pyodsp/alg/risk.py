"""How a program values a distribution of outcomes.

The default is the expectation, which is what "stochastic programming"
means unqualified: minimize the average cost. A risk measure changes that
— CVaR weights the bad tail, so the solution pays something in expectation
to protect against the outcomes that hurt.

These live here rather than with the modelling front-end because the
bundle method builds them into the master problem, and pyodsp.alg cannot
depend on pyodsp.model.

Everything here is stated for a *minimization*: larger is worse, and the
tail of interest is the upper one. That is the only convention the
algorithms ever see — PyomoSolver converts a maximize model on
construction — and it means one formulation covers both senses. For a
maximize program the same upper tail of the converted objective is the
lower tail of the original, which is the one a risk-averse modeller
wants.
"""

from dataclasses import dataclass
from typing import Sequence

# Below this much probability in the tail, CVaR is the worst outcome and
# the 1/(1-alpha) factor stops being usable.
MIN_TAIL = 1e-9


@dataclass(frozen=True)
class Expectation:
    """Value a distribution by its mean. The risk-neutral default."""

    @property
    def is_risk_neutral(self) -> bool:
        return True

    def describe(self) -> str:
        return "expectation"


@dataclass(frozen=True)
class CVaR:
    """Conditional Value at Risk, blended with the expectation.

    The objective becomes

        (1 - weight) * E[Z]  +  weight * CVaR_alpha[Z]

    where CVaR_alpha[Z] is the mean of the worst (1 - alpha) of outcomes.

    Attributes:
        alpha: The confidence level. 0.95 attends to the worst 5% of
            outcomes; 0 makes CVaR the plain expectation, and values near
            1 make it the single worst outcome.
        weight: How much of the objective the tail accounts for. 0 is
            exactly risk-neutral — which is worth using as a check, since
            it must reproduce the expectation solution.
    """

    alpha: float = 0.95
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError(
                f"alpha must be in [0, 1), got {self.alpha}. At 1 the tail is "
                "empty and CVaR is undefined; use a value just below it for "
                "the worst outcome alone."
            )
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(
                f"weight must be in [0, 1], got {self.weight}. It is the share "
                "of the objective the tail accounts for, the rest being the "
                "expectation."
            )

    @property
    def is_risk_neutral(self) -> bool:
        return self.weight == 0.0 or self.alpha == 0.0

    @property
    def tail(self) -> float:
        """The probability mass CVaR averages over."""
        return 1.0 - self.alpha

    def describe(self) -> str:
        return f"CVaR(alpha={self.alpha:g}, weight={self.weight:g})"


RiskMeasure = Expectation | CVaR


def cvar_of_sample(
    values: Sequence[float], probabilities: Sequence[float], alpha: float
) -> float:
    """CVaR of a discrete distribution, computed exactly.

    Walks the outcomes worst-first and averages until the tail's
    probability mass is used up, splitting the outcome that straddles the
    boundary. That is the value the Rockafellar-Uryasev program in the
    master converges to, so it is what the incumbent must be measured
    with — an incumbent computed as an expectation would be compared
    against a risk-adjusted bound and the two would never meet.

    Args:
        values: The outcomes, in the minimize convention.
        probabilities: Their probabilities, summing to one.
        alpha: The confidence level.
    """
    if not values:
        raise ValueError("No outcomes to take the CVaR of")
    tail = 1.0 - alpha
    if tail <= MIN_TAIL:
        return max(values)

    worst_first = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    taken = 0.0
    total = 0.0
    for i in worst_first:
        share = min(probabilities[i], tail - taken)
        if share <= 0.0:
            break
        total += share * values[i]
        taken += share
        if taken >= tail:
            break
    # `taken` can fall short only if the probabilities do not sum to one,
    # which ScenarioSet already rejects; dividing by it rather than by the
    # tail keeps the answer a mean either way.
    return total / taken if taken > 0.0 else max(values)


def value_of_sample(
    risk: RiskMeasure, values: Sequence[float], probabilities: Sequence[float]
) -> float:
    """What `risk` makes of this distribution, in the minimize convention."""
    expectation = sum(p * v for p, v in zip(probabilities, values))
    if isinstance(risk, Expectation) or risk.is_risk_neutral:
        return expectation
    tail_value = cvar_of_sample(values, probabilities, risk.alpha)
    return (1.0 - risk.weight) * expectation + risk.weight * tail_value

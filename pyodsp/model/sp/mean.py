"""Collapsing a scenario set to its mean realization.

The expected-value problem replaces the uncertain data with its
expectation, so something has to average a `Scenario.data` — which the
pipeline otherwise never inspects, since it belongs to the recourse
builder. The rule here is deliberately narrow: probability-weight
anything numeric, require anything else to be the same in every
scenario, and refuse the rest by name rather than guess.
"""

from typing import Any, Dict, List, Sequence

from .scenario import Scenario, ScenarioSet

MEAN_SCENARIO_NAME = "mean"


def _is_number(value: Any) -> bool:
    # bool is a subclass of int, and the mean of True and False is not a
    # flag — averaging one would be silently meaningless.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _describe(path: Sequence[Any]) -> str:
    return "data" + "".join(f"[{part!r}]" for part in path)


def _average(values: List[Any], weights: Sequence[float], path: Sequence[Any]) -> Any:
    """One entry of the mean scenario, from its value in each scenario."""
    first = values[0]

    if _is_number(first):
        if not all(_is_number(v) for v in values):
            raise ValueError(
                f"Cannot average {_describe(path)}: it is a number in some "
                "scenarios and not in others."
            )
        return sum(weight * value for weight, value in zip(weights, values))

    if isinstance(first, dict):
        keys = set(first)
        for value in values:
            if not isinstance(value, dict) or set(value) != keys:
                raise ValueError(
                    f"Cannot average {_describe(path)}: the scenarios give it "
                    "different keys. Every scenario must describe the same "
                    "quantities."
                )
        return {
            key: _average([value[key] for value in values], weights, (*path, key))
            for key in first
        }

    if isinstance(first, (list, tuple)):
        length = len(first)
        for value in values:
            if not isinstance(value, (list, tuple)) or len(value) != length:
                raise ValueError(
                    f"Cannot average {_describe(path)}: the scenarios give it "
                    "different lengths."
                )
        averaged = [
            _average([value[i] for value in values], weights, (*path, i))
            for i in range(length)
        ]
        return type(first)(averaged) if isinstance(first, tuple) else averaged

    if hasattr(first, "shape") and hasattr(first, "__mul__"):
        # numpy arrays and anything else that scales and adds elementwise
        shapes = {getattr(value, "shape", None) for value in values}
        if len(shapes) > 1:
            raise ValueError(
                f"Cannot average {_describe(path)}: the scenarios give it "
                f"different shapes ({sorted(map(str, shapes))})."
            )
        total = weights[0] * values[0]
        for weight, value in zip(weights[1:], values[1:]):
            total = total + weight * value
        return total

    # Not numeric. It can still pass through unchanged if it never varies —
    # a shared label or lookup table is data the scenarios agree on.
    if all(value == first for value in values[1:]):
        return first
    raise ValueError(
        f"Cannot average {_describe(path)}: it is not numeric and differs "
        f"between scenarios ({first!r} vs "
        f"{next(v for v in values[1:] if v != first)!r}). Pass "
        "mean_scenario=... to say what its mean realization is."
    )


def mean_scenario(
    scenarios: ScenarioSet, override: Dict[str, Any] | None = None
) -> Scenario:
    """The single scenario whose data is the expectation of the set's.

    Args:
        scenarios: The scenarios to average.
        override: Entries to use instead of the averaged ones. Supplying
            every key skips the averaging entirely, which is the way out
            for data that cannot be averaged meaningfully.

    Returns:
        A Scenario of probability 1 — the expected-value problem is
        deterministic, and this is its only realization.
    """
    weights = list(scenarios.probabilities)
    override = dict(override or {})

    keys = set(scenarios[0].data)
    for scenario in scenarios:
        if set(scenario.data) != keys:
            raise ValueError(
                f"Scenario {scenario.name!r} describes "
                f"{sorted(scenario.data)}, but {scenarios[0].name!r} describes "
                f"{sorted(keys)}. Every scenario must describe the same "
                "quantities before they can be averaged."
            )

    unknown = set(override) - keys
    if unknown:
        raise ValueError(
            f"mean_scenario overrides {sorted(unknown)}, which no scenario "
            f"has. The scenarios describe {sorted(keys)}."
        )

    data = {
        key: override[key]
        if key in override
        else _average([s.data[key] for s in scenarios], weights, (key,))
        for key in keys
    }
    return Scenario(MEAN_SCENARIO_NAME, 1.0, data)

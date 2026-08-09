"""Scenarios for a two-stage stochastic program.

This module only covers scenarios given as explicit data — a finite list
of realizations with probabilities. Sampling and reduction helpers, if
they arrive, are expected to produce a ScenarioSet and hand it here.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

PROBABILITY_SUM_TOL = 1e-9


@dataclass(frozen=True)
class Scenario:
    """One realization of the uncertain data.

    Attributes:
        name: Identifies the scenario in logs, results and plots.
        probability: Its probability. Probabilities over a ScenarioSet
            sum to one.
        data: The realized values, handed to the recourse builder as-is.
    """

    name: str
    probability: float
    data: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getattr__(self, key: str) -> Any:
        # Only reached for names that are not real attributes, so the
        # dataclass fields above keep priority over scenario data.
        try:
            return self.data[key]
        except KeyError:
            raise AttributeError(
                f"Scenario {self.name!r} has no data entry {key!r}; "
                f"it has {sorted(self.data)}"
            ) from None


class ScenarioSet(Sequence[Scenario]):
    """An ordered, probability-normalized collection of Scenarios.

    Order is meaningful: it fixes the leaf node indices, so a rerun of the
    same ScenarioSet writes its per-scenario output to the same places.
    """

    def __init__(self, scenarios: Iterable[Scenario], *, normalize: bool = False):
        self._scenarios: List[Scenario] = list(scenarios)
        if not self._scenarios:
            raise ValueError("A stochastic program needs at least one scenario")

        names = [s.name for s in self._scenarios]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"Duplicate scenario names: {sorted(duplicates)}")

        for s in self._scenarios:
            if s.probability < 0.0:
                raise ValueError(
                    f"Scenario {s.name!r} has negative probability {s.probability}"
                )

        total = sum(s.probability for s in self._scenarios)
        if total <= 0.0:
            raise ValueError("Scenario probabilities must sum to a positive number")
        if normalize:
            self._scenarios = [
                Scenario(s.name, s.probability / total, s.data) for s in self._scenarios
            ]
        elif abs(total - 1.0) > PROBABILITY_SUM_TOL:
            raise ValueError(
                f"Scenario probabilities sum to {total}, not 1. Pass "
                "normalize=True to rescale them, or supply weights that "
                "already sum to one."
            )

    def __len__(self) -> int:
        return len(self._scenarios)

    def __getitem__(self, index):  # type: ignore[override]
        if isinstance(index, slice):
            return ScenarioSet(self._scenarios[index])
        return self._scenarios[index]

    def __iter__(self) -> Iterator[Scenario]:
        return iter(self._scenarios)

    def __repr__(self) -> str:
        return f"ScenarioSet({[s.name for s in self._scenarios]!r})"

    @property
    def names(self) -> List[str]:
        return [s.name for s in self._scenarios]

    @property
    def probabilities(self) -> List[float]:
        return [s.probability for s in self._scenarios]

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        name_key: str = "name",
        probability_key: str = "probability",
        normalize: bool = False,
    ) -> "ScenarioSet":
        """Build from a list of dicts, one per scenario.

        Keys other than `name_key` and `probability_key` become the
        scenario's data. A record with no `name_key` is named by its
        position; with no `probability_key`, scenarios are equally likely.

        Args:
            records: The per-scenario mappings.
            name_key: Key holding the scenario name.
            probability_key: Key holding the scenario probability.
            normalize: Rescale probabilities to sum to one rather than
                rejecting a set that does not.
        """
        records = list(records)
        if not records:
            raise ValueError("No scenario records given")

        missing_probability = [probability_key not in r for r in records]
        if any(missing_probability) and not all(missing_probability):
            raise ValueError(
                f"Either every record carries {probability_key!r} or none does; "
                "a partial set has no sensible completion."
            )
        uniform = all(missing_probability)

        scenarios = []
        for position, record in enumerate(records):
            data = {
                k: v for k, v in record.items() if k not in (name_key, probability_key)
            }
            name = str(record.get(name_key, position))
            probability = (
                1.0 / len(records) if uniform else float(record[probability_key])
            )
            scenarios.append(Scenario(name, probability, data))
        return cls(scenarios, normalize=normalize)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Mapping[str, Any]],
        probabilities: Mapping[str, float] | None = None,
        *,
        normalize: bool = False,
    ) -> "ScenarioSet":
        """Build from {scenario name: scenario data}.

        Args:
            data: Per-scenario data, keyed by scenario name.
            probabilities: Per-scenario probability, keyed the same way.
                Equally likely scenarios if omitted.
            normalize: Rescale probabilities to sum to one.
        """
        if not data:
            raise ValueError("No scenario data given")
        if probabilities is None:
            probabilities = {name: 1.0 / len(data) for name in data}
        missing = set(data) - set(probabilities)
        if missing:
            raise ValueError(f"No probability given for scenarios {sorted(missing)}")
        extra = set(probabilities) - set(data)
        if extra:
            raise ValueError(f"Probability given for unknown scenarios {sorted(extra)}")
        return cls(
            [
                Scenario(name, float(probabilities[name]), dict(d))
                for name, d in data.items()
            ],
            normalize=normalize,
        )

    @classmethod
    def from_dataframe(
        cls,
        frame,
        *,
        name_column: str = "name",
        probability_column: str = "probability",
        normalize: bool = False,
    ) -> "ScenarioSet":
        """Build from a pandas DataFrame with one row per scenario.

        Every column other than `name_column` and `probability_column`
        becomes a data entry of that scenario.
        """
        return cls.from_records(
            frame.to_dict(orient="records"),
            name_key=name_column,
            probability_key=probability_column,
            normalize=normalize,
        )


def as_scenario_set(scenarios, *, normalize: bool = False) -> ScenarioSet:
    """Coerce the accepted user-facing spellings into a ScenarioSet."""
    if isinstance(scenarios, ScenarioSet):
        return scenarios
    if hasattr(scenarios, "to_dict") and hasattr(scenarios, "columns"):
        return ScenarioSet.from_dataframe(scenarios, normalize=normalize)
    if isinstance(scenarios, Mapping):
        return ScenarioSet.from_mapping(scenarios, normalize=normalize)

    scenarios = list(scenarios)
    if all(isinstance(s, Scenario) for s in scenarios):
        return ScenarioSet(scenarios, normalize=normalize)
    if all(isinstance(s, Mapping) for s in scenarios):
        return ScenarioSet.from_records(scenarios, normalize=normalize)
    raise TypeError(
        "scenarios must be a ScenarioSet, a DataFrame, a mapping of "
        "name -> data, or an iterable of Scenario or of dicts"
    )

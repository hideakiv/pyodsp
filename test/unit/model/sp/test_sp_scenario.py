import pandas as pd
import pytest

from pyodsp.model.scenario import (
    Scenario,
    ScenarioSet,
    as_scenario_set,
)


def test_records_keep_their_probabilities_and_order():
    scenarios = ScenarioSet.from_records(
        [
            {"name": "lo", "probability": 0.25, "demand": 3},
            {"name": "hi", "probability": 0.75, "demand": 9},
        ]
    )

    assert scenarios.names == ["lo", "hi"]
    assert scenarios.probabilities == [0.25, 0.75]
    assert scenarios[0].data == {"demand": 3}


def test_records_without_probabilities_are_equally_likely():
    scenarios = ScenarioSet.from_records([{"demand": d} for d in (1, 2, 3, 4)])

    assert scenarios.probabilities == [0.25] * 4
    # unnamed records fall back to their position
    assert scenarios.names == ["0", "1", "2", "3"]


def test_partial_probabilities_are_rejected():
    with pytest.raises(ValueError, match="every record carries"):
        ScenarioSet.from_records([{"name": "a", "probability": 0.5}, {"name": "b"}])


def test_probabilities_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to"):
        ScenarioSet.from_records(
            [
                {"name": "a", "probability": 0.3},
                {"name": "b", "probability": 0.3},
            ]
        )


def test_normalize_rescales_instead_of_rejecting():
    scenarios = ScenarioSet.from_records(
        [
            {"name": "a", "probability": 3.0},
            {"name": "b", "probability": 1.0},
        ],
        normalize=True,
    )

    assert scenarios.probabilities == [0.75, 0.25]


def test_duplicate_names_are_rejected():
    with pytest.raises(ValueError, match="Duplicate scenario names"):
        ScenarioSet([Scenario("a", 0.5), Scenario("a", 0.5)])


def test_negative_probability_is_rejected():
    with pytest.raises(ValueError, match="negative probability"):
        ScenarioSet([Scenario("a", -0.5), Scenario("b", 1.5)])


def test_an_empty_set_is_rejected():
    with pytest.raises(ValueError, match="at least one scenario"):
        ScenarioSet([])


def test_scenario_data_is_reachable_three_ways():
    scenario = Scenario("s", 1.0, {"demand": 7, "price": 2.0})

    assert scenario["demand"] == 7
    assert scenario.demand == 7
    assert scenario.data["demand"] == 7
    assert scenario.get("missing", "fallback") == "fallback"


def test_unknown_data_key_names_what_is_available():
    scenario = Scenario("s", 1.0, {"demand": 7})

    with pytest.raises(AttributeError, match="demand"):
        scenario.supply


def test_dataclass_fields_win_over_data_of_the_same_name():
    scenario = Scenario("s", 0.5, {"name": "shadow", "probability": 99})

    assert scenario.name == "s"
    assert scenario.probability == 0.5


def test_mapping_defaults_to_uniform_probabilities():
    scenarios = ScenarioSet.from_mapping({"a": {"d": 1}, "b": {"d": 2}})

    assert scenarios.probabilities == [0.5, 0.5]


def test_mapping_rejects_a_probability_for_an_unknown_scenario():
    with pytest.raises(ValueError, match="unknown scenarios"):
        ScenarioSet.from_mapping({"a": {"d": 1}}, {"a": 0.5, "ghost": 0.5})


def test_mapping_rejects_a_missing_probability():
    with pytest.raises(ValueError, match="No probability given"):
        ScenarioSet.from_mapping({"a": {"d": 1}, "b": {"d": 2}}, {"a": 1.0})


def test_dataframe_columns_become_scenario_data():
    frame = pd.DataFrame(
        {"name": ["a", "b"], "probability": [0.4, 0.6], "demand": [1, 2]}
    )

    scenarios = ScenarioSet.from_dataframe(frame)

    assert scenarios.names == ["a", "b"]
    assert scenarios[1].data == {"demand": 2}


@pytest.mark.parametrize(
    "given",
    [
        [{"name": "a", "probability": 0.5}, {"name": "b", "probability": 0.5}],
        [Scenario("a", 0.5), Scenario("b", 0.5)],
        {"a": {}, "b": {}},
        pd.DataFrame({"name": ["a", "b"], "probability": [0.5, 0.5]}),
    ],
)
def test_every_accepted_spelling_coerces(given):
    assert as_scenario_set(given).names == ["a", "b"]


def test_an_unusable_spelling_is_rejected():
    with pytest.raises(TypeError, match="scenarios must be"):
        as_scenario_set([1, 2, 3])

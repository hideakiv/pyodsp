"""Averaging a scenario set down to its mean realization."""

import numpy as np
import pytest

from pyodsp.model.sp.mean import mean_scenario
from pyodsp.model.scenario import Scenario, ScenarioSet


def make(*data, probabilities=None):
    probabilities = probabilities or [1.0 / len(data)] * len(data)
    return ScenarioSet(
        [Scenario(str(i), p, d) for i, (d, p) in enumerate(zip(data, probabilities))]
    )


def test_a_flat_number_is_probability_weighted():
    scenarios = make({"d": 3.0}, {"d": 7.0}, probabilities=[0.25, 0.75])

    assert mean_scenario(scenarios).data == {"d": 6.0}


def test_weighting_is_by_probability_not_a_plain_average():
    # The distinction only shows on an uneven set, and getting it wrong
    # would give the wrong expected-value problem silently.
    scenarios = make({"d": 0.0}, {"d": 10.0}, probabilities=[0.9, 0.1])

    assert mean_scenario(scenarios).data["d"] == pytest.approx(1.0)


def test_nested_dicts_are_averaged_entry_by_entry():
    scenarios = make(
        {"yield": {"WHEAT": 3.0, "CORN": 3.6}},
        {"yield": {"WHEAT": 2.0, "CORN": 2.4}},
    )

    assert mean_scenario(scenarios).data == {"yield": {"WHEAT": 2.5, "CORN": 3.0}}


def test_lists_are_averaged_elementwise_and_keep_their_type():
    scenarios = make({"path": [1.0, 2.0]}, {"path": [3.0, 4.0]})

    averaged = mean_scenario(scenarios).data["path"]

    assert averaged == [2.0, 3.0]
    assert isinstance(averaged, list)


def test_tuples_stay_tuples():
    scenarios = make({"pair": (1.0, 2.0)}, {"pair": (3.0, 4.0)})

    assert mean_scenario(scenarios).data["pair"] == (2.0, 3.0)


def test_numpy_arrays_are_averaged():
    scenarios = make({"a": np.array([1.0, 2.0])}, {"a": np.array([3.0, 6.0])})

    assert list(mean_scenario(scenarios).data["a"]) == [2.0, 4.0]


def test_deep_nesting_is_followed():
    scenarios = make(
        {"a": {"b": [{"c": 0.0}]}},
        {"a": {"b": [{"c": 4.0}]}},
    )

    assert mean_scenario(scenarios).data == {"a": {"b": [{"c": 2.0}]}}


def test_a_constant_non_numeric_entry_passes_through():
    scenarios = make({"label": "winter", "d": 1.0}, {"label": "winter", "d": 3.0})

    assert mean_scenario(scenarios).data == {"label": "winter", "d": 2.0}


def test_a_varying_non_numeric_entry_is_refused_by_name():
    scenarios = make({"regime": "wet"}, {"regime": "dry"})

    with pytest.raises(ValueError, match=r"data\['regime'\].*not numeric"):
        mean_scenario(scenarios)


def test_booleans_are_not_treated_as_numbers():
    # The mean of True and False is not a flag; averaging one would be
    # meaningless rather than merely imprecise.
    scenarios = make({"open": True}, {"open": False})

    with pytest.raises(ValueError, match=r"data\['open'\]"):
        mean_scenario(scenarios)


def test_a_constant_boolean_passes_through():
    scenarios = make({"open": True, "d": 1.0}, {"open": True, "d": 3.0})

    assert mean_scenario(scenarios).data["open"] is True


def test_mismatched_nested_keys_are_refused():
    scenarios = make({"y": {"A": 1.0}}, {"y": {"B": 1.0}})

    with pytest.raises(ValueError, match="different keys"):
        mean_scenario(scenarios)


def test_mismatched_list_lengths_are_refused():
    scenarios = make({"p": [1.0]}, {"p": [1.0, 2.0]})

    with pytest.raises(ValueError, match="different lengths"):
        mean_scenario(scenarios)


def test_scenarios_describing_different_quantities_are_refused():
    scenarios = make({"d": 1.0}, {"e": 1.0})

    with pytest.raises(ValueError, match="must describe the same"):
        mean_scenario(scenarios)


def test_an_override_replaces_the_averaged_entry():
    scenarios = make({"regime": "wet", "d": 1.0}, {"regime": "dry", "d": 3.0})

    averaged = mean_scenario(scenarios, {"regime": "normal"})

    assert averaged.data == {"regime": "normal", "d": 2.0}


def test_an_override_for_an_unknown_key_is_refused():
    scenarios = make({"d": 1.0}, {"d": 3.0})

    with pytest.raises(ValueError, match="which no scenario has"):
        mean_scenario(scenarios, {"ghost": 1.0})


def test_the_mean_scenario_is_certain():
    # The expected-value problem is deterministic: one realization, p = 1.
    scenarios = make({"d": 1.0}, {"d": 3.0})

    assert mean_scenario(scenarios).probability == 1.0

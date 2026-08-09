from pyodsp.dec.dd.mip_heuristic_root import aggregate_final_up_messages
from pyodsp.dec.dd.message import DdFinalUpMessage


def test_aggregates_objectives_by_summing():
    messages = {1: DdFinalUpMessage(2.0), 2: DdFinalUpMessage(3.0)}

    result = aggregate_final_up_messages(messages)

    assert result.get_objective() == 5.0


def test_returns_none_objective_when_any_child_is_none():
    messages = {1: DdFinalUpMessage(2.0), 2: DdFinalUpMessage(None)}

    result = aggregate_final_up_messages(messages)

    assert result.get_objective() is None


def test_returns_zero_for_empty_messages():
    result = aggregate_final_up_messages({})

    assert result.get_objective() == 0.0

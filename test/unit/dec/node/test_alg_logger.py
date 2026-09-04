import logging

from pyodsp.dec.node._logger import AlgLogger


def messages(caplog):
    return [record.getMessage() for record in caplog.records]


def test_records_go_to_the_channel_logger(caplog):
    caplog.set_level(logging.INFO)

    AlgLogger("Benders decomposition", "bd").log_info("hello")

    assert caplog.records[0].name == "pyodsp.dec.bd"
    assert messages(caplog) == ["hello"]


def test_initialization_names_the_method_and_lists_kwargs(caplog):
    caplog.set_level(logging.INFO)

    AlgLogger("Dual decomposition", "dd").log_initialization(tolerance=1e-6)

    assert messages(caplog) == [
        "Starting Dual decomposition",
        "tolerance: 1e-06",
    ]


def test_completion_is_a_single_line_with_the_objective(caplog):
    caplog.set_level(logging.INFO)

    AlgLogger("SDDP", "sddp").log_completion(-855.8333333333)

    assert messages(caplog) == [
        "SDDP completed. Final objective value: -855.8333333333"
    ]


def test_completion_without_an_objective_omits_the_value(caplog):
    caplog.set_level(logging.INFO)

    AlgLogger("Dual decomposition", "dd").log_completion(None)

    assert messages(caplog) == ["Dual decomposition completed"]


def test_level_none_leaves_the_logger_to_inherit():
    logging.getLogger("pyodsp.dec.bdsc").setLevel(logging.WARNING)

    AlgLogger("Benders decomposition with scaled cuts", "bdsc", level=None)

    assert logging.getLogger("pyodsp.dec.bdsc").level == logging.WARNING

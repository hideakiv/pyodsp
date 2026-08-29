import logging

from pyodsp.alg.bm.logger import BmLogger, render_iteration
from pyodsp import _PyodspFormatter


def make_logger(node_id, depth, level=logging.INFO):
    return BmLogger("Test Method", node_id, depth, level)


def records(caplog):
    return caplog.records


# --- records carry structured context, not a baked-in prefix ---------------


def test_message_is_clean_and_context_rides_alongside(caplog):
    caplog.set_level(logging.DEBUG)

    make_logger("root", 0).log_info("hello")

    (record,) = records(caplog)
    assert record.getMessage() == "hello"
    assert record.name == "pyodsp.alg.bm.root"
    assert record.pyodsp == {"node_id": "root", "depth": 0}


def test_debug_output_still_needs_the_level(caplog):
    caplog.set_level(logging.DEBUG)

    make_logger("dbg", 2, level=logging.DEBUG).log_debug("detail")

    (record,) = records(caplog)
    assert record.getMessage() == "detail"
    assert record.levelno == logging.DEBUG


def test_initialization_names_the_method_and_lists_kwargs(caplog):
    caplog.set_level(logging.DEBUG)

    make_logger("init", 1).log_initialization(tolerance=1e-6)

    assert [r.getMessage() for r in records(caplog)] == [
        "Starting Test Method",
        "tolerance: 1e-06",
    ]
    assert all(r.pyodsp == {"node_id": "init", "depth": 1} for r in records(caplog))


def test_iteration_row_carries_its_fields(caplog):
    caplog.set_level(logging.INFO)

    make_logger("it", 1).log_iteration(
        iteration=3, lb=-889.5, ub=-248.47, num_cuts=3, elapsed=0.03
    )

    (record,) = records(caplog)
    assert record.getMessage() == (
        "Iteration: 3\tLB: -889.5000\tUB: -248.4700\tNumCuts: 3\tElapsed: 0.03"
    )
    assert record.pyodsp["iteration"] == {
        "iteration": 3,
        "lb": -889.5,
        "ub": -248.47,
        "num_cuts": 3,
        "elapsed": 0.03,
    }


# --- render_iteration: one row builder, optional columns -------------------


def test_render_iteration_plain():
    assert render_iteration(
        {"iteration": 1, "lb": None, "ub": None, "num_cuts": 0, "elapsed": 0.01}
    ) == "Iteration: 1\tLB: -\tUB: -\tNumCuts: 0\tElapsed: 0.01"


def test_render_iteration_with_center_and_penalty():
    row = render_iteration(
        {
            "iteration": 2,
            "lb": -2299.2,
            "center_bound": None,
            "ub": -470.4,
            "num_cuts": 1,
            "penalty": 5.0,
            "elapsed": 0.02,
        }
    )
    assert row == (
        "Iteration: 2\tLB: -2299.2000\tCB: -\tUB: -470.4000\t"
        "NumCuts: 1\tu: 5.0\tElapsed: 0.02"
    )


# --- the formatter turns context into a tree prefix -----------------------


def _format(record):
    return _PyodspFormatter("%(pyodsp_prefix)s%(message)s").format(record)


def test_formatter_prefixes_root_flush_left(caplog):
    caplog.set_level(logging.INFO)
    make_logger("root", 0).log_info("hello")
    assert _format(caplog.records[0]) == "Node: root - hello"


def test_formatter_indents_by_depth(caplog):
    caplog.set_level(logging.INFO)
    make_logger("deep", 3).log_info("hello")
    assert _format(caplog.records[0]) == "      Node: deep - hello"


def test_formatter_tolerates_a_negative_depth(caplog):
    caplog.set_level(logging.INFO)
    make_logger("odd", -1).log_info("hello")
    assert _format(caplog.records[0]) == "Node: odd - hello"


def test_formatter_leaves_context_free_records_unprefixed():
    record = logging.LogRecord(
        "pyodsp.dec.bd", logging.INFO, __file__, 1, "Starting Benders decomposition", (), None
    )
    assert _format(record) == "Starting Benders decomposition"

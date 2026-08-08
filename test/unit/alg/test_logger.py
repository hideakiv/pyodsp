import logging

import pytest

from pyodsp.alg.bm.logger import BmLogger, INDENT


def make_logger(node_id, depth, level=logging.INFO):
    return BmLogger("Test Method", node_id, depth, level)


def messages(caplog):
    return [record.getMessage() for record in caplog.records]


def test_root_output_is_flush_left(caplog):
    caplog.set_level(logging.DEBUG)

    make_logger("root", 0).log_info("hello")

    assert messages(caplog) == ["Node: root - hello"]


def test_each_level_steps_in(caplog):
    caplog.set_level(logging.DEBUG)

    make_logger("one", 1).log_info("hello")
    make_logger("two", 2).log_info("hello")

    assert messages(caplog) == [
        f"{INDENT}Node: one - hello",
        f"{INDENT * 2}Node: two - hello",
    ]


def test_debug_output_is_indented_too(caplog):
    caplog.set_level(logging.DEBUG)

    # the logger's own level gates debug output, independently of capture
    make_logger("dbg", 2, level=logging.DEBUG).log_debug("detail")

    assert messages(caplog) == [f"{INDENT * 2}Node: dbg - detail"]


def test_initialization_is_indented_like_everything_else(caplog):
    caplog.set_level(logging.DEBUG)

    make_logger("init", 1).log_initialization(tolerance=1e-6)

    assert messages(caplog) == [
        f"{INDENT}Node: init - Starting Test Method",
        f"{INDENT}Node: init - tolerance: 1e-06",
    ]


def test_status_lines_are_indented(caplog):
    caplog.set_level(logging.DEBUG)

    logger = make_logger("status", 3)
    logger.log_status_optimal()
    logger.log_completion(iteration=4, objective_value=1.5)

    assert messages(caplog) == [
        f"{INDENT * 3}Node: status - Test Method terminated by optimality",
        f"{INDENT * 3}Node: status - Test Method completed",
        f"{INDENT * 3}Node: status - Total iterations: 4",
        f"{INDENT * 3}Node: status - Final objective value: 1.5",
    ]


def test_a_negative_depth_does_not_break_formatting(caplog):
    caplog.set_level(logging.DEBUG)

    make_logger("odd", -1).log_info("hello")

    assert messages(caplog) == ["Node: odd - hello"]

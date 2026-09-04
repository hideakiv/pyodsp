import argparse

import pyodsp


def get_args():
    # pyodsp emits log records but installs no handler; opt in so the example
    # prints the algorithm's progress.
    pyodsp.configure_logging()
    parser = argparse.ArgumentParser(description="bd examples")
    parser.add_argument(
        "--solver", type=str, default="appsi_highs", help="solver for pyomo"
    )

    args = parser.parse_args()
    return args


def assert_approximately_equal(a, b, tolerance=1e-3):
    assert abs(a - b) <= tolerance, (
        f"{a} and {b} are not approximately equal within {tolerance}"
    )

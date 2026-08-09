"""The multistage example runs and reports a sensible optimum."""

import subprocess
import sys
from pathlib import Path

import pytest

# Example scripts are named relative to the repo root, and they write their
# output under output/ relative to the cwd, so run them from there.
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("stages", ["3", "4"])
def test_the_example_runs(stages):
    result = subprocess.run(
        [sys.executable, "examples/inventory/msp_pipeline.py", "--stages", stages],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "via sddp" in result.stdout
    assert "the high-demand path:" in result.stdout


@pytest.mark.parametrize("ranks", ["2", "4"])
def test_the_mpi_example_agrees_with_the_serial_one(ranks):
    """The answer must not depend on how many ranks ran it.

    Every rank builds the identical lattice and rank 0 drives the
    iteration; the others only help evaluate the Monte Carlo sample, whose
    draws are keyed by global sample index precisely so the result is
    rank-count independent (see LatticeMpi).
    """
    serial = subprocess.run(
        [sys.executable, "examples/inventory/msp_pipeline.py", "--stages", "3"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert serial.returncode == 0, serial.stderr[-2000:]

    parallel = subprocess.run(
        [
            "mpiexec",
            "-n",
            ranks,
            sys.executable,
            "examples/inventory/msp_pipeline_mpi.py",
            "--stages",
            "3",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=600,
    )
    assert parallel.returncode == 0, parallel.stderr[-2000:]

    assert _bound(serial.stdout) == pytest.approx(_bound(parallel.stdout), abs=1e-6)


def test_only_one_rank_reports():
    # Every rank returns a result; only rank 0's carries the answer, and
    # the example is expected to guard on that rather than print four
    # copies of it.
    parallel = subprocess.run(
        [
            "mpiexec",
            "-n",
            "3",
            sys.executable,
            "examples/inventory/msp_pipeline_mpi.py",
            "--stages",
            "3",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=600,
    )

    assert parallel.returncode == 0, parallel.stderr[-2000:]
    assert parallel.stdout.count("via sddp") == 1


def _bound(output: str) -> float:
    for line in output.splitlines():
        if "bound" in line and ":" in line:
            return float(line.split(":")[-1])
    raise AssertionError(f"no bound in output:\n{output[-2000:]}")

"""The farmer examples all solve the same problem four ways.

bd_model.py in particular went unnoticed while broken — it fed a maximize
model straight into an algorithm that rejects one — so it is covered here
alongside the rest. Each script asserts its own optimum internally; a
non-zero exit means the answer moved.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Example scripts are named relative to the repo root, and they write their
# output under output/ relative to the cwd, so run them from there.
ROOT = Path(__file__).resolve().parents[3]

SCRIPTS = [
    "examples/farmer/model.py",
    "examples/farmer/stochastic_model.py",
    "examples/farmer/bd_model.py",
    "examples/farmer/sp_pipeline.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_the_example_runs(script):
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize("method", ["bd", "dd"])
def test_the_pipeline_example_agrees_across_algorithms(method):
    result = subprocess.run(
        [sys.executable, "examples/farmer/sp_pipeline.py", "--method", method],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "objective : 108390" in result.stdout

"""The time-dependent-uncertainty example runs and behaves as advertised."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

# Example scripts are named relative to the repo root, and they write their
# output under output/ relative to the cwd, so run them from there.
ROOT = Path(__file__).resolve().parents[3]
RHO = 0.6
SCRIPT = "examples/hydro/msp_time_dependent.py"


@pytest.fixture(scope="module")
def output():
    result = subprocess.run(
        [sys.executable, SCRIPT],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def test_it_solves_and_reports_both_branches(output):
    assert "via sddp" in output
    assert "wettest outcome each season" in output
    assert "driest outcome each season" in output


def test_the_lattice_changes_shape_with_the_stage(output):
    # spring 2, summer 3, autumn 2, winter 2 — the distribution and the
    # number of realizations both vary
    assert "nodes per stage : [1, 2, 3, 2, 2]" in output


def _inflows(output: str, heading: str):
    """The inflow column of one reported path."""
    block = output.split(heading)[1].split("\n\n")[0]
    rows = [line.split() for line in block.strip().splitlines()[1:]]
    return [float(row[2]) for row in rows]


@pytest.mark.parametrize(
    "heading, noise",
    [
        ("wettest outcome each season:", [30.0, 20.0, 24.0, 34.0]),
        ("driest outcome each season:", [22.0, 0.0, 12.0, 26.0]),
    ],
)
def test_the_inflows_follow_the_autoregressive_recursion(output, heading, noise):
    """The lag is genuinely carried, not just declared.

    Each stage's inflow has to be RHO times the previous one plus that
    season's noise — which only holds if the state augmentation is
    actually transporting the value between stages.
    """
    inflows = _inflows(output, heading)

    assert len(inflows) == 5
    for previous, realized, drawn in zip(inflows, inflows[1:], noise):
        assert realized == pytest.approx(RHO * previous + drawn, abs=0.05)


def test_releases_rise_with_the_price_when_water_is_scarce(output):
    """The policy holds water back for the stages that pay best.

    On the dry branch the reservoir cannot serve every stage at capacity,
    so what it does with the water it has is the whole decision — and it
    should shift it towards the expensive stages.
    """
    block = output.split("driest outcome each season:")[1]
    rows = [line.split() for line in block.strip().splitlines()[1:6]]
    prices = [float(row[1]) for row in rows]
    releases = [float(row[3]) for row in rows]

    # skip the first stage, which starts from a full reservoir
    later_prices, later_releases = prices[1:], releases[1:]
    assert later_prices == sorted(later_prices), "prices should be rising"
    assert later_releases == sorted(
        later_releases
    ), f"releases {later_releases} should rise with prices {later_prices}"


def test_the_example_is_self_describing(output):
    assert re.search(r"state variables\s*:\s*2", output)


def test_the_plot_flag_writes_every_figure(tmp_path):
    result = subprocess.run(
        [sys.executable, SCRIPT, "--plot"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    written = {
        Path(line.split("wrote ", 1)[1]).name
        for line in result.stdout.splitlines()
        if line.startswith("wrote ")
    }
    assert written == {
        "convergence.png",
        "scenario_lattice.png",
        "objective_distribution.png",
        "trajectory_storage.png",
        "trajectory_inflow.png",
    }


def test_it_reports_the_objective_statistics(output):
    assert "simulated objective over the sampled paths:" in output
    for name in ("count", "mean", "std", "ci_lower", "ci_upper", "bound"):
        assert name in output

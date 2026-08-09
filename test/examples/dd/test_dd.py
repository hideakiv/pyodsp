import subprocess
import sys
from pathlib import Path

# Example scripts are named relative to the repo root, and they write their
# output under output/ relative to the cwd, so run them from there.
ROOT = Path(__file__).resolve().parents[3]

solvers = ["appsi_highs"]


def test_equality():
    for solver in solvers:
        result = subprocess.run(
            [sys.executable, "examples/dd/equality.py", "--solver", solver],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0


def test_equality_pbm():
    for solver in solvers:
        result = subprocess.run(
            [sys.executable, "examples/dd/equality_pbm.py", "--solver", solver],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0


def test_ray():
    for solver in solvers:
        result = subprocess.run(
            [sys.executable, "examples/dd/ray.py", "--solver", solver],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0


def test_equality_mpi():
    for solver in solvers:
        result = subprocess.run(
            [
                "mpiexec",
                "-n",
                "4",
                sys.executable,
                "examples/dd/equality_mpi.py",
                "--solver",
                solver,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0


def test_equality_mip():
    for solver in solvers:
        result = subprocess.run(
            [sys.executable, "examples/dd/equality_mip.py", "--solver", solver],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0


def test_equality_mip_mpi():
    for solver in solvers:
        result = subprocess.run(
            [
                "mpiexec",
                "-n",
                "4",
                sys.executable,
                "examples/dd/equality_mip_mpi.py",
                "--solver",
                solver,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0

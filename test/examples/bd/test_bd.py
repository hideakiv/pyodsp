import subprocess

solvers = ["appsi_highs"]


def test_feasibility():
    for solver in solvers:
        result = subprocess.run(
            ["python", "examples/bd/feasibility.py", "--solver", solver],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


def test_optimality():
    for solver in solvers:
        result = subprocess.run(
            ["python", "examples/bd/optimality.py", "--solver", solver],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


def test_optimality_mpi():
    for solver in solvers:
        result = subprocess.run(
            [
                "mpiexec",
                "-n",
                "3",
                "python",
                "examples/bd/optimality_mpi.py",
                "--solver",
                solver,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


def test_aircon():
    for solver in solvers:
        result = subprocess.run(
            ["python", "examples/aircon/bd.py", "--solver", solver],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

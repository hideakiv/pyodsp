import subprocess

solvers = ["appsi_highs"]


def test_equality():
    for solver in solvers:
        result = subprocess.run(
            ["python", "examples/dd/equality.py", "--solver", solver],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


def test_equality_pbm():
    for solver in solvers:
        result = subprocess.run(
            ["python", "examples/dd/equality_pbm.py", "--solver", solver],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


def test_ray():
    for solver in solvers:
        result = subprocess.run(
            ["python", "examples/dd/ray.py", "--solver", solver],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


def test_equality_mpi():
    for solver in solvers:
        result = subprocess.run(
            [
                "mpiexec",
                "-n",
                "4",
                "python",
                "examples/dd/equality_mpi.py",
                "--solver",
                solver,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


def test_equality_mip():
    for solver in solvers:
        result = subprocess.run(
            ["python", "examples/dd/equality_mip.py", "--solver", solver],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


def test_equality_mip_mpi():
    for solver in solvers:
        result = subprocess.run(
            [
                "mpiexec",
                "-n",
                "4",
                "python",
                "examples/dd/equality_mip_mpi.py",
                "--solver",
                solver,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

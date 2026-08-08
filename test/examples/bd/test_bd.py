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


def test_sddp():
    for solver in solvers:
        result = subprocess.run(
            ["python", "examples/aircon/sddp.py", "--solver", solver],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


def test_sddp_mpi():
    result = subprocess.run(
        ["mpiexec", "-n", "3", "python", "examples/aircon/sddp_mpi.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_bdsc():
    for solver in solvers:
        result = subprocess.run(
            ["python", "examples/bdsc/cs.py", "--solver", solver],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


def test_bdsc_mpi():
    result = subprocess.run(
        ["mpiexec", "-n", "3", "python", "examples/bdsc/cs_mpi.py"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0


def test_bdsc_mpi_with_fewer_ranks_than_scenarios():
    # one rank takes both scenarios
    result = subprocess.run(
        ["mpiexec", "-n", "2", "python", "examples/bdsc/cs_mpi.py"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0


def test_bdsc_mpi_with_more_ranks_than_scenarios():
    # rank 3 is left holding nothing; it used to block forever waiting for
    # an init message the root only sent to ranks that owned leaves
    result = subprocess.run(
        ["mpiexec", "-n", "4", "python", "examples/bdsc/cs_mpi.py"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0

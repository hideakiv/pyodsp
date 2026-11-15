"""SDDP with scaled cuts under MPI: the answer must not depend on how many
ranks ran it.

As for plain SDDP (test/examples/inventory/test_inventory.py), every rank
builds the identical lattice, rank 0 runs every forward and backward pass,
and the others only evaluate their slice of the Monte Carlo sample. Here a
replica is sent each master's rho along with its cuts (see
pyodsp.dec.sddpsc.run_mpi). sddpsc_mpi_script.py solves the same models
the serial tests do; the end-to-end tests launch it under mpiexec.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyomo.environ as pyo
import pytest

from pyodsp.alg.bm.cuts import OptimalityCut, CutList
from pyodsp.dec.sddpsc.alg_root_bm import SddpScAlgRootBm
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig

SCRIPT = Path(__file__).resolve().parent / "sddpsc_mpi_script.py"
SOLVER = "appsi_highs"

needs_mpiexec = pytest.mark.skipif(
    shutil.which("mpiexec") is None, reason="needs mpiexec"
)


def solve(model, output_dir, ranks=None):
    command = [sys.executable, str(SCRIPT), model, str(output_dir)]
    if ranks is not None:
        command = ["mpiexec", "-n", str(ranks), *command, "--mpi"]
    run = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
    # only rank 0 reports
    assert len(lines) == 1, run.stdout[-2000:]
    return json.loads(lines[0])


@needs_mpiexec
@pytest.mark.parametrize(
    "model, ranks",
    [("cover", 2), ("cover", 3), ("markov", 3)],
    ids=["integer-2", "integer-3", "markov-3"],
)
def test_mpi_agrees_with_the_serial_run(tmp_path, model, ranks):
    serial = solve(model, tmp_path / "serial")
    parallel = solve(model, tmp_path / "mpi", ranks=ranks)

    assert parallel["bound"] == pytest.approx(serial["bound"], abs=1e-6)
    assert parallel["first_stage"] == pytest.approx(serial["first_stage"])
    # the sample draws are keyed by global index, so the ranks between
    # them evaluate exactly the serial run's paths
    assert parallel["samples"] == pytest.approx(serial["samples"])
    assert "via sddpsc" in parallel["summary"]


# -- what a replica is sent ---------------------------------------------------


def master(purgeable=True):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, 2))
    model.obj = pyo.Objective(expr=model.x)
    alg = SddpScAlgRootBm(
        PyomoSolver(model, SolverConfig(SOLVER), [model.x]), purgeable=purgeable
    )
    alg.set_logger(1, 1, 0)
    alg.build([[10, 11]], {10: 0.5, 11: 0.5}, {10: -5.0, 11: -5.0})
    return alg


@pytest.mark.parametrize("cuts_changed", [False, True])
def test_a_replica_steps_exactly_as_rank_0_would(cuts_changed):
    # cuts_changed=False is a visit where rho advanced but no cut was
    # taken: the next step keeps rho as is, so it has to have travelled.
    # True is one where a cut was taken: the next step resyncs rho from
    # the cuts on both.
    source = master()
    source.run_step(None)
    cut = OptimalityCut(coeffs={0: 1.0}, rhs=-2.0, info={}, objective_value=-2.0)
    source.bm.add_cuts([CutList([cut])])
    source.rho += 0.75  # as add_cuts' rho step leaves it
    source._cuts_changed = cuts_changed

    replica = master(purgeable=False)
    replica.apply_snapshot(source.get_snapshot())

    assert replica.rho == source.rho
    assert replica.bm.cpm.get_num_cuts() == source.bm.cpm.get_num_cuts() == 1
    _, expected = source.run_step(None)
    _, got = replica.run_step(None)
    assert got.get_rho() == pytest.approx(expected.get_rho())
    assert got.get_solution() == pytest.approx(expected.get_solution())
    if not cuts_changed:
        assert got.get_rho() != pytest.approx(master().rho)


def test_the_snapshot_is_a_copy():
    source = master()
    snapshot = source.get_snapshot()
    source.rho += 1.0
    assert snapshot["rho"] != source.rho


# -- the front-end -----------------------------------------------------------


def test_the_front_end_picks_the_mpi_runner(tmp_path):
    from pyodsp.dec.sddpsc.run_mpi import SddpScRunMpi
    from test_sddpsc_msp import ONE, store

    assert store(tmp_path, ONE, mpi=True)._runner() is SddpScRunMpi


def test_a_replica_rank_builds_masters_that_do_not_purge(tmp_path, monkeypatch):
    from pyodsp.alg.bm.cuts_manager import NonPurgingCutsManager
    from pyodsp.model.msp import problem as problem_module
    from test_sddpsc_msp import ONE, store

    monkeypatch.setattr(
        problem_module.MultistageProgram, "rank", property(lambda self: 2)
    )
    built = store(tmp_path, ONE, mpi=True).build()

    for stage in built.nodes[:-1]:
        for node in stage:
            manager = node.alg_root.bm.cpm.cuts_manager
            assert isinstance(manager, NonPurgingCutsManager)

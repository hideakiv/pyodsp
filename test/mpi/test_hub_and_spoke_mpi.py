"""MPI-level tests for HubAndSpokeMpi.

Unlike the rest of the suite, this module needs genuinely separate OS
processes coordinating over MPI collectives (bcast/gather/send/recv) — a
single-process pytest run cannot exercise the non-root branches at all.
It is a no-op (skipped) under plain `pytest`, and only does real work when
launched with at least 3 MPI ranks:

    mpiexec -n 3 python -m pytest test/mpi/test_hub_and_spoke_mpi.py -v

Each rank builds only the one node it owns (mirroring how
examples/dd/equality_mpi.py splits nodes across ranks) and only rank 0
makes assertions about the aggregated result — a non-zero exit on any
rank aborts the whole mpiexec run, so other ranks must not raise.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit" / "dec"))
from fakes import (
    FakeAlgRoot,
    FakeAlgLeaf,
    FakeLogger,
    FakeInitDnMessage,
    FakeInitUpMessage,
    FakeUpMessage,
    FakeDnMessage,
    FakeFinalDnMessage,
    FakeFinalUpMessage,
)

from pyodsp.alg.bm.cuts import OptimalityCut
from pyodsp.alg.const import STATUS_NOT_FINISHED, STATUS_OPTIMAL
from pyodsp.dec.graph.hub_and_spoke_mpi import HubAndSpokeMpi
from pyodsp.dec.node.dec_node import DecNodeParent, DecNodeChild

REQUIRED_RANKS = 3


def test_run_splits_work_across_ranks(tmp_path=None):
    comm = MPI.COMM_WORLD
    if comm.Get_size() < REQUIRED_RANKS:
        pytest.skip(
            f"requires mpiexec -n {REQUIRED_RANKS}; "
            f"got size={comm.Get_size()} (see module docstring)"
        )

    rank = comm.Get_rank()
    # every rank needs the same directory; tmp_path isn't shared across
    # separate mpiexec-launched processes, so derive one deterministically.
    filedir = Path("/tmp") / "pyodsp_mpi_test_hub_and_spoke"

    cut = OptimalityCut(coeffs={0: 1.0}, rhs=1.0, info={}, objective_value=1.0)

    if rank == 0:
        alg_root = FakeAlgRoot()
        alg_root.get_init_dn_message = MagicMock(
            return_value=FakeInitDnMessage(is_minimize=True)
        )
        alg_root.get_final_dn_message = MagicMock(return_value=FakeFinalDnMessage())
        alg_root.pass_final_up_message = MagicMock(
            return_value=FakeFinalUpMessage(objective=42.0)
        )
        root = DecNodeParent(idx=0, alg_root=alg_root)
        root.add_child(1)
        root.add_child(2)

        dn = FakeDnMessage(objective=10.0)
        alg_root.run_step = MagicMock(
            side_effect=[(STATUS_NOT_FINISHED, dn), (STATUS_OPTIMAL, dn)]
        )

        nodes = [root]
    else:
        alg_leaf = FakeAlgLeaf()
        alg_leaf.get_init_up_message = MagicMock(return_value=FakeInitUpMessage())
        alg_leaf.get_up_message = MagicMock(return_value=FakeUpMessage(cut=cut))
        alg_leaf.get_final_up_message = MagicMock(
            return_value=FakeFinalUpMessage(objective=1.0)
        )
        leaf = DecNodeChild(idx=rank, alg_leaf=alg_leaf)
        nodes = [leaf]

    graph = HubAndSpokeMpi(nodes, FakeLogger(), filedir=filedir, max_iteration=10)
    graph.run()

    if rank == 0:
        assert alg_root.run_step.call_count == 2
        alg_root.pass_final_up_message.assert_called_once()

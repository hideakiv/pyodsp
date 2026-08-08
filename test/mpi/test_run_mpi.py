"""MPI-level tests for BdRunMpi/DdRunMpi — thin wrappers around HubAndSpokeMpi.

Same constraint as test_hub_and_spoke_mpi.py: needs real MPI ranks. Run with:

    mpiexec -n 3 python -m pytest test/mpi/test_run_mpi.py -v
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
    FakeInitDnMessage,
    FakeInitUpMessage,
    FakeUpMessage,
    FakeFinalDnMessage,
    FakeFinalUpMessage,
)

from pyodsp.alg.bm.cuts import OptimalityCut
from pyodsp.alg.const import STATUS_OPTIMAL
from pyodsp.dec.bd.run_mpi import BdRunMpi
from pyodsp.dec.bd.message import BdDnMessage
from pyodsp.dec.node.dec_node import DecNodeParent, DecNodeChild

REQUIRED_RANKS = 2


def test_bd_run_mpi_completes_across_ranks():
    comm = MPI.COMM_WORLD
    if comm.Get_size() < REQUIRED_RANKS:
        pytest.skip(
            f"requires mpiexec -n {REQUIRED_RANKS}; "
            f"got size={comm.Get_size()} (see module docstring)"
        )

    rank = comm.Get_rank()
    filedir = Path("/tmp") / "pyodsp_mpi_test_run_mpi"
    cut = OptimalityCut(coeffs={0: 1.0}, rhs=1.0, info={}, objective_value=1.0)

    if rank == 0:
        alg_root = FakeAlgRoot()
        alg_root.get_init_dn_message = MagicMock(
            return_value=FakeInitDnMessage(is_minimize=True)
        )
        alg_root.get_final_dn_message = MagicMock(return_value=FakeFinalDnMessage())
        alg_root.pass_final_up_message = MagicMock(
            return_value=FakeFinalUpMessage(objective=7.0)
        )
        root = DecNodeParent(idx=0, alg_root=alg_root)
        root.add_child(1)
        alg_root.run_step = MagicMock(return_value=(STATUS_OPTIMAL, BdDnMessage([0.0])))

        nodes = [root]
    else:
        alg_leaf = FakeAlgLeaf()
        alg_leaf.get_init_up_message = MagicMock(return_value=FakeInitUpMessage())
        alg_leaf.get_up_message = MagicMock(return_value=FakeUpMessage(cut=cut))
        alg_leaf.get_final_up_message = MagicMock(
            return_value=FakeFinalUpMessage(objective=1.0)
        )
        nodes = [DecNodeChild(idx=1, alg_leaf=alg_leaf)]

    bd_run = BdRunMpi(nodes, filedir, max_iteration=10)
    bd_run.run()

    if rank == 0:
        alg_root.pass_final_up_message.assert_called_once()

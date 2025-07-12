from pathlib import Path
from mpi4py import MPI

from dd import create_master, create_sub
from pyodsp.dec.dd.run_mpi import DdRunMpi

"""
mpiexec -n 6 python dd_mpi.py
"""


def main(
    N: list[int],
    P: int,
    d: list[int],
    L: list[int],
    c: list[float],
    l: list[int],
    solver="appsi_highs",
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    K = len(N)

    assert size == K + 1

    if rank == 0:
        node = create_master(N, P, d, solver)
        for k in range(K):
            node.add_child(k + 1)
    else:
        k = rank - 1
        node = create_sub(k, N[k], P, L[k], c[k], l, solver)

    dd_run = DdRunMpi([node], Path("output/mcsp/dd_mpi"))
    dd_run.run()


if __name__ == "__main__":
    pass

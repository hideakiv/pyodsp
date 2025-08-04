from pathlib import Path
from mpi4py import MPI

from dd import create_master, create_sub
from params import McspParams, create_single, create_random
from pyodsp.dec.dd.run_mpi import DdRunMpi

"""
mpiexec -n 6 python dd_mpi.py
"""


def main(param: McspParams, solver="appsi_highs"):
    K = param.K
    P = param.P
    N = param.N
    d = param.d
    L = param.L
    c = param.c
    l = param.l
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
    K = 5
    P = 8
    param = create_random(K, P)
    main(param)

from pathlib import Path
from mpi4py import MPI

from dd import create_master, create_sub
from params import UcParams
from pyodsp.dec.dd.run_mpi import DdRunMpi

"""
mpiexec -n 6 python dd_mpi.py
"""


def main(
    num_time: int,
    num_gens: int,
    demand: list[float],
    params: dict[int, UcParams],
    solver="appsi_highs",
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert size == num_gens + 1

    if rank == 0:
        node = create_master(num_time, num_gens, demand, solver, pbm=True)
        for k in range(1, num_gens):
            node.add_child(k)
    else:
        k = rank
        node = create_sub(k, num_time, params, solver)

    dd_run = DdRunMpi([node], Path("output/uc/dd_mpi"))
    dd_run.run()


if __name__ == "__main__":
    pass

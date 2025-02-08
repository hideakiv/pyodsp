from mpi4py import MPI

from equality import create_master, create_sub
from pyodec.dec.dd.run_mpi import DdRunMpi

"""
mpiexec -n 4 python equality_mpi.py
"""


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        node = create_master()
        node.add_child(1)
        node.add_child(2)
        node.add_child(3)
    if rank == 1:
        node = create_sub(1)
    if rank == 2:
        node = create_sub(2)
    if rank == 3:
        node = create_sub(3)

    node_rank_map = {0: 0, 1: 1, 2: 2, 3: 3}

    dd_run = DdRunMpi([node], node_rank_map)
    dd_run.run()

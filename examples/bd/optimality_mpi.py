from pathlib import Path
from mpi4py import MPI

from optimality import create_root_node, create_leaf_node, p
from pyodec.dec.bd.run_mpi import BdRunMpi


"""
mpiexec -n 3 python optimality_mpi.py
"""


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        node = create_root_node()
        node.add_child(1, multiplier=p[1])
        node.add_child(2, multiplier=p[2])

    if rank == 1:
        node = create_leaf_node(1)

    if rank == 2:
        node = create_leaf_node(2)

    node_rank_map = {0: 0, 1: 1, 2: 2}

    bd_run = BdRunMpi([node], node_rank_map, Path("output/bd/optimality_mpi"))
    bd_run.run()

    MPI.Finalize()


if __name__ == "__main__":
    main()

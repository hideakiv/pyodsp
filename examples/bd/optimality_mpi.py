from pathlib import Path
from mpi4py import MPI

from optimality import create_root_node, create_leaf_node, p
from pyodsp.dec.bd.run_mpi import BdRunMpi

from utils import get_args, assert_approximately_equal

"""
mpiexec -n 3 python optimality_mpi.py
"""


def main():
    args = get_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        node = create_root_node(args.solver)
        node.add_child(1, multiplier=p[1])
        node.add_child(2, multiplier=p[2])

    if rank == 1:
        node = create_leaf_node(1, args.solver)

    if rank == 2:
        node = create_leaf_node(2, args.solver)

    node_rank_map = {0: 0, 1: 1, 2: 2}

    bd_run = BdRunMpi([node], node_rank_map, Path("output/bd/optimality_mpi"))
    bd_run.run()

    if rank == 0:
        assert_approximately_equal(node.alg.bm.obj_bound[-1], -855.83333333333)

    MPI.Finalize()


if __name__ == "__main__":
    main()

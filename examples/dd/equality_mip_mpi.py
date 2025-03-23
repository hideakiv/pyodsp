from pathlib import Path
from mpi4py import MPI

from equality_mip import create_master, create_sub
from pyodsp.dec.dd.run_mpi import DdRunMpi

from utils import get_args, assert_approximately_equal

"""
mpiexec -n 4 python equality_mpi.py
"""


def main():
    args = get_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        node = create_master(args.solver)
        node.add_child(1)
        node.add_child(2)
        node.add_child(3)
    if rank == 1:
        node = create_sub(1, args.solver)
    if rank == 2:
        node = create_sub(2, args.solver)
    if rank == 3:
        node = create_sub(3, args.solver)

    dd_run = DdRunMpi([node], Path("output/dd/equality_mip_mpi"))
    dd_run.run()

    if rank == 0:
        assert_approximately_equal(node.alg_root.bm.obj_bound[-1], -19.666666666666664)
    if rank == 1:
        assert_approximately_equal(node.alg_leaf.solver.get_solution()[0], 1.0)
    if rank == 2:
        assert_approximately_equal(node.alg_leaf.solver.get_solution()[0], 2.0)
    if rank == 3:
        assert_approximately_equal(node.alg_leaf.solver.get_solution()[0], 2.0)


if __name__ == "__main__":
    main()

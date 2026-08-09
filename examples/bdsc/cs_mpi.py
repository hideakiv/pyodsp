from pathlib import Path

from mpi4py import MPI

from cs import create_root_node, create_leaf_node
from pyodsp.dec.bdsc.run_mpi import BdScRunMpi

from utils import get_args, assert_approximately_equal

"""
mpiexec -n 3 python cs_mpi.py

Rank 0 holds the root, and the scenarios are dealt round robin over the
remaining ranks — so this runs on 2 ranks (one taking every scenario),
on R + 1 (a rank each), or on more than that, where the ranks past the
scenarios simply have nothing to do. The root's children have to be a
single group, which it declares even though the leaves themselves live on
other ranks.
"""

R = 2


def main():
    args = get_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        raise SystemExit(
            "run with at least 2 ranks: rank 0 takes the root and the "
            f"other ranks share the {R} scenarios"
        )

    nodes = []
    if rank == 0:
        root = create_root_node(R, args.solver)
        for s in range(1, R + 1):
            root.add_child(s, multiplier=1 / R)
        root.set_groups([[s for s in range(1, R + 1)]])
        nodes.append(root)

    for s in range(1, R + 1):
        if (s - 1) % (size - 1) + 1 == rank:
            nodes.append(create_leaf_node(s, R, args.solver))

    bd_run = BdScRunMpi(nodes, Path("output/bdsc/cs_mpi"))
    bd_run.run()

    if rank == 0:
        assert_approximately_equal(nodes[0].alg_root.bm.obj_bound[-1], 0.2031)

    MPI.Finalize()


if __name__ == "__main__":
    main()

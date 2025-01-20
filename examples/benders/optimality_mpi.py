from mpi4py import MPI

from .optimality import create_root_node, create_leaf_node, p

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        node = create_root_node()
        node.add_child(1, multiplier=p[1])
        node.add_child(2, multiplier=p[2])

    if rank == 1:
        node = create_leaf_node(1)

    if rank == 2:
        node = create_leaf_node(2)

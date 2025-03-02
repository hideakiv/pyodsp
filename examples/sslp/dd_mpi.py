from pathlib import Path
from mpi4py import MPI

from dd import create_master, create_sub
from pyodsp.dec.dd.run_mpi import DdRunMpi

"""
mpiexec -n 6 python dd_mpi.py
"""

def main(nI: int, nJ: int, nS: int, solver="appsi_highs"):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert size == nS + 1

    if rank == 0:
        node = create_master(nJ, nS, solver)
        for s in range(nS):
            node.add_child(s+1)
    else:
        s = rank - 1
        node = create_sub(s, nI, nJ, nS, solver)

    dd_run = DdRunMpi([node], Path("output/sslp/dd_mpi"))
    dd_run.run()

if __name__ == "__main__":
    main(50, 10, 5)
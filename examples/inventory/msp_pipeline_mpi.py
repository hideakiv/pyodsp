"""The multistage inventory example, with the simulation across MPI ranks.

    mpiexec -n 4 python examples/inventory/msp_pipeline_mpi.py

Identical to msp_pipeline.py but for `mpi=True`. Under MPI every rank
builds the same lattice — unlike bd's and dd's MPI runners, which split
distinct nodes across ranks — and rank 0 drives the iteration while the
others help evaluate the Monte Carlo sample that tests convergence. Only
rank 0 writes output and holds the answer, which is what
`result.is_root_rank` is for.
"""

import argparse

import pyodsp
from msp_pipeline import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="appsi_highs")
    parser.add_argument("--stages", type=int, default=4)
    args = parser.parse_args()

    # pyodsp emits log records but installs no handler; opt in to see progress.
    pyodsp.configure_logging()

    msp = build(args.stages, args.solver, mpi=True)

    result = msp.solve()

    # Every rank returns a result; only one of them has anything in it.
    if result.is_root_rank:
        print(result.summary())


if __name__ == "__main__":
    main()

"""Solve one of the sddpsc test models, serially or under MPI, and print
what test_sddpsc_mpi.py compares: the bound, the first stage, and the last
round's simulated costs.

    python test/solver/sddpsc_mpi_script.py cover OUTDIR
    mpiexec -n 3 python test/solver/sddpsc_mpi_script.py cover OUTDIR --mpi
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_sddpsc_msp import TWO, store  # noqa: E402
from test_sddpsc_regression import cover  # noqa: E402

MARKOV = [[[0.5, 0.5]], [[0.2, 0.8], [0.8, 0.2]]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["cover", "markov"])
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mpi", action="store_true")
    args = parser.parse_args()

    if args.model == "cover":
        msp = cover(args.output_dir, "sddpsc", mpi=args.mpi)
    else:
        msp = store(args.output_dir, TWO, transitions=MARKOV, mpi=args.mpi)

    result = msp.solve()
    if not result.is_root_rank:
        return
    print(
        json.dumps(
            {
                "bound": result.bound,
                "first_stage": result.first_stage_flat,
                "samples": sorted(float(v) for v in result.simulation_samples),
                "stop_reason": result.stop_reason,
                "summary": result.summary(),
            }
        )
    )


if __name__ == "__main__":
    main()

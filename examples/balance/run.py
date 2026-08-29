import argparse
import numpy as np
import pyodsp
from lp import solve_lp
from sddp import solve_sddp
from scenarios import create_static_scenarios
from regime import (
    NormalRegime,
    HotSunnyRegime,
    HotCloudyRegime,
    ColdSunnyRegime,
    ColdCloudyRegime,
    RegimeParams,
)


def main(mode: str, num_stages: int):
    # pyodsp emits log records but installs no handler; opt in to see progress.
    pyodsp.configure_logging()
    time = 48
    r1 = NormalRegime(time, 42)
    r2 = HotSunnyRegime(time, 43)
    r3 = HotCloudyRegime(time, 44)
    r4 = ColdSunnyRegime(time, 45)
    r5 = ColdCloudyRegime(time, 46)
    regimes = [r1, r2, r3, r4, r5]
    tm = np.asarray(
        [
            [0.6, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.6, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.6, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.6],
        ]
    )
    regime_params = RegimeParams(regimes, tm)
    params = create_static_scenarios(num_stages, [1, 1, 1, 1, 1], regime_params, 0)
    if mode == "lp":
        build_time, solve_time = solve_lp(params)
    elif mode == "sddp":
        build_time, solve_time = solve_sddp(params)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    print(f"Build time: {build_time:.2f} seconds")
    print(f"Solve time: {solve_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the balance example with LP or SDDP."
    )
    parser.add_argument(
        "mode",
        choices=["lp", "sddp"],
        help="Solve mode: lp or sddp",
    )
    parser.add_argument(
        "num_stages",
        type=int,
        help="Number of stages to build scenarios for",
    )
    args = parser.parse_args()

    main(args.mode, args.num_stages)

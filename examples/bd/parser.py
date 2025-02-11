import argparse


def get_args():
    parser = argparse.ArgumentParser(description="bd examples")
    parser.add_argument(
        "--solver", type=str, default="appsi_highs", help="solver for pyomo"
    )

    args = parser.parse_args()
    return args

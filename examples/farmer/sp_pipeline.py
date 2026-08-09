"""Birge and Louveaux's farmer, through the two-stage pipeline.

Compare with bd_model.py, which wires the same problem to pyodsp.dec by
hand: there the first-stage variables are re-declared inside every
scenario, the two coupling lists are built in matching order by hand,
each leaf is given an invented bound, and the maximize objective has to
be negated on every model before the algorithms will accept it. None of
that appears below.

    python examples/farmer/sp_pipeline.py
    python examples/farmer/sp_pipeline.py --method dd --plot
"""

import argparse

import pyomo.environ as pyo

from pyodsp.model.sp import StochasticProgram

CROPS = ["WHEAT", "CORN", "BEETS"]
TOTAL_ACREAGE = 500

PLANTING_COST = {"WHEAT": 150.0, "CORN": 230.0, "BEETS": 260.0}
SELLING_PRICE = {"WHEAT": 170.0, "CORN": 150.0, "BEETS": 36.0}
OVER_QUOTA_PRICE = {"WHEAT": 0.0, "CORN": 0.0, "BEETS": 10.0}
PRICE_QUOTA = {"WHEAT": 100000.0, "CORN": 100000.0, "BEETS": 6000.0}
CATTLE_FEED = {"WHEAT": 200.0, "CORN": 240.0, "BEETS": 0.0}
PURCHASE_PRICE = {"WHEAT": 238.0, "CORN": 210.0, "BEETS": 100000.0}

YIELDS = {
    "GOOD": {"WHEAT": 3.0, "CORN": 3.6, "BEETS": 24.0},
    "AVERAGE": {"WHEAT": 2.5, "CORN": 3.0, "BEETS": 20.0},
    "POOR": {"WHEAT": 2.0, "CORN": 2.4, "BEETS": 16.0},
}

OPTIMAL_PROFIT = 108390.0


def build(solver: str, method: str) -> StochasticProgram:
    sp = StochasticProgram(
        "farmer",
        sense="max",
        method=method,
        solver=solver,
        output_dir="output/farmer/sp_pipeline",
    )

    @sp.first_stage
    def first_stage(model):
        """How many acres to devote to each crop, before the yield is known."""
        model.acreage = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
        model.land = pyo.Constraint(
            expr=sum(model.acreage[c] for c in CROPS) <= TOTAL_ACREAGE
        )
        return -sum(PLANTING_COST[c] * model.acreage[c] for c in CROPS)

    @sp.recourse
    def recourse(model, state, scenario):
        """What to sell and buy once this scenario's yield is realized.

        `state.acreage` is this scenario's own copy of the first-stage
        decision; the pipeline creates it and couples it.
        """
        harvest = scenario["yield"]

        model.sold = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
        model.sold_over_quota = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
        model.kept = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
        model.purchased = pyo.Var(CROPS, domain=pyo.NonNegativeReals)

        model.harvest_balance = pyo.Constraint(
            CROPS,
            rule=lambda m, c: m.sold[c] + m.sold_over_quota[c] + m.kept[c]
            == harvest[c] * state.acreage[c],
        )
        model.feed_requirement = pyo.Constraint(
            CROPS, rule=lambda m, c: m.kept[c] + m.purchased[c] >= CATTLE_FEED[c]
        )
        model.quota = pyo.Constraint(
            CROPS, rule=lambda m, c: m.sold[c] <= PRICE_QUOTA[c]
        )

        return (
            sum(SELLING_PRICE[c] * model.sold[c] for c in CROPS)
            + sum(OVER_QUOTA_PRICE[c] * model.sold_over_quota[c] for c in CROPS)
            - sum(PURCHASE_PRICE[c] * model.purchased[c] for c in CROPS)
        )

    # Equally likely by default; pass probabilities to weight them.
    sp.set_scenarios({name: {"yield": y} for name, y in YIELDS.items()})
    return sp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="appsi_highs")
    parser.add_argument("--method", default="bd", choices=["bd", "dd", "bdsc"])
    parser.add_argument(
        "--plot", action="store_true", help="write charts (needs matplotlib)"
    )
    args = parser.parse_args()

    sp = build(args.solver, args.method)
    print(sp.describe(), end="\n\n")

    result = sp.solve()
    print(result.summary())

    if args.plot:
        for path in result.plot():
            print(f"wrote {path}")

    assert (
        abs(result.objective - OPTIMAL_PROFIT) < 1e-3
    ), f"expected {OPTIMAL_PROFIT}, got {result.objective}"


if __name__ == "__main__":
    main()

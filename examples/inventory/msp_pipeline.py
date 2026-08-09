"""A multistage inventory problem, through the multistage pipeline.

Each stage you may buy stock, then a demand arrives. Buying early is
cheap but commits you before you know what the demand will be; buying
late is dearer but informed. The state carried between stages is the
inventory left over.

Compare with examples/balance/sddp.py, which wires the same kind of
problem to pyodsp.dec by hand: there every node's model, both coupling
lists, the node ids, the transition probabilities and the per-node bounds
are built explicitly. Here one builder describes a stage and the pipeline
does the rest.

    python examples/inventory/msp_pipeline.py
    python examples/inventory/msp_pipeline.py --stages 6 --plot
"""

import argparse

import pyomo.environ as pyo

from pyodsp.model.msp import MultistageProgram

BUY_NOW = 2.0
BUY_LATER = 5.0
SHORTAGE_PENALTY = 20.0
CAPACITY = 200.0

# What demand may turn out to be, at every stage after the first.
REALIZATIONS = [
    {"name": "low", "probability": 0.3, "demand": 10.0},
    {"name": "mid", "probability": 0.5, "demand": 25.0},
    {"name": "high", "probability": 0.2, "demand": 40.0},
]


def build(stages: int, solver: str, mpi: bool = False) -> MultistageProgram:
    msp = MultistageProgram(
        "inventory",
        sense="min",
        solver=solver,
        # No stage can cost less than nothing, which is the floor every
        # stage's cost-to-go needs before its parent can price it.
        stage_bound=0.0,
        output_dir="output/inventory/msp_pipeline",
        sample_frequency=10,
        sample_size=50,
        mpi=mpi,
    )

    @msp.stage(state=["inventory"])
    def stage(model, state, node):
        """One stage: buy, meet what demand you can, carry the rest.

        `state.inventory` is what the previous stage left — a number at
        stage 0, where it is the initial condition, and a variable
        everywhere else. `model.inventory` is what this stage leaves.
        """
        model.inventory = pyo.Var(bounds=(0, CAPACITY))
        model.bought = pyo.Var(bounds=(0, CAPACITY))
        model.shortage = pyo.Var(bounds=(0, CAPACITY))

        demand = node.get("demand", 0.0)
        model.balance = pyo.Constraint(
            expr=model.inventory
            == state.inventory + model.bought - demand + model.shortage
        )

        price = BUY_NOW if node.is_first else BUY_LATER
        return price * model.bought + SHORTAGE_PENALTY * model.shortage

    msp.set_initial_state(inventory=0.0)
    msp.set_realizations(REALIZATIONS, stages=stages)
    return msp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="appsi_highs")
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument(
        "--plot", action="store_true", help="write the convergence chart"
    )
    args = parser.parse_args()

    msp = build(args.stages, args.solver)
    print(msp.describe(), end="\n\n")

    result = msp.solve()
    print(result.summary(), end="\n\n")

    # SDDP leaves a policy, not a schedule: any path can be replayed
    # against the cuts afterwards. Follow the high-demand branch.
    worst = [f"{stage}-{len(REALIZATIONS) - 1}" for stage in range(1, args.stages)]
    print("the high-demand path:")
    for step in result.simulate(["0-0", *worst]):
        carried = step.next_state[0] if step.next_state else 0.0
        print(
            f"  {step.node_idx}: cost {step.stage_cost:8.2f}   carried {carried:6.2f}"
        )

    if args.plot:
        print(f"\nwrote {result.plot_convergence()}")


if __name__ == "__main__":
    main()

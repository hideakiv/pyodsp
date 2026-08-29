"""Hydro reservoir scheduling with uncertainty that changes over time.

A reservoir earns money by releasing water through a turbine. How much
water arrives is uncertain, and the problem is when to release it: water
held back is available for a later, better-paid stage, but the reservoir
can overflow, and water spilled earns nothing.

The point of the example is that all three kinds of time dependence show
up in one model, and each is handled differently:

1. Deterministic data that varies with the stage — the electricity price.
   `node.stage` indexes it. Nothing special is needed.

2. A distribution that varies with the stage — inflows are wetter in
   spring than in summer. `set_stage_realizations` takes one set of
   realizations per stage, so the distribution (and the number of
   realizations) may differ at each.

3. A process that depends on its own past — inflows are autocorrelated:
   a dry month tends to follow a dry month. SDDP needs the uncertainty to
   be stage-wise independent or Markov, so a process with memory is
   handled by *augmenting the state* with the lag. `inflow` is carried
   between stages exactly like `storage` is, and this stage's inflow is

       inflow_t = RHO * inflow_{t-1} + noise_t

   The dependence has to stay affine in the carried value: SDDP's cuts
   are only valid where the value function is convex in the state, and a
   multiplicative lag would break that silently.

    python examples/hydro/msp_time_dependent.py
    python examples/hydro/msp_time_dependent.py --plot
"""

import argparse

import pyomo.environ as pyo

import pyodsp
from pyodsp.model.msp import MultistageProgram

CAPACITY = 200.0
TURBINE = 60.0
INITIAL_STORAGE = 100.0
INITIAL_INFLOW = 40.0

# How strongly this month's inflow remembers last month's.
RHO = 0.6

# (1) deterministic, stage-varying: power is worth most in winter.
PRICE = {0: 20.0, 1: 18.0, 2: 25.0, 3: 34.0, 4: 52.0}

# (2) stage-varying distributions: the seasons the reservoir passes
# through, each with its own inflow noise. Spring runs wet and reliable;
# summer is dry and much more variable; winter is wet again.
SEASONS = {
    1: ("spring", [(0.5, 30.0), (0.5, 22.0)]),
    2: ("summer", [(0.2, 20.0), (0.5, 8.0), (0.3, 0.0)]),
    3: ("autumn", [(0.4, 24.0), (0.6, 12.0)]),
    4: ("winter", [(0.6, 34.0), (0.4, 26.0)]),
}


# Every node needs a ceiling on its value-to-go before its parent can
# price it. A stage cannot earn more than a full turbine at that stage's
# price, so a node is worth at most that summed over the stages it still
# has ahead of it. One bound covers them all, so it is the earliest
# node's — stage 1, with the whole horizon left.
#
# Worth keeping tight: a loose ceiling is the master's first estimate of
# the future, so it is where the bound starts and how far it has to
# travel. The obvious slack alternative, best price x turbine x every
# stage, starts it at 15,600 instead of 7,740 and buries the interesting
# part of the convergence plot.
STAGE_BOUND = TURBINE * sum(price for stage, price in PRICE.items() if stage > 0)


def realizations_for(stage: int):
    season, outcomes = SEASONS[stage]
    return [
        {"name": f"{season}-{i}", "probability": probability, "noise": noise}
        for i, (probability, noise) in enumerate(outcomes)
    ]


def build(solver: str) -> MultistageProgram:
    msp = MultistageProgram(
        "hydro",
        sense="max",
        solver=solver,
        stage_bound=STAGE_BOUND,
        output_dir="output/hydro/msp_time_dependent",
        sample_frequency=10,
        sample_size=100,
    )

    @msp.stage(state=["storage", "inflow"])
    def stage(model, state, node):
        """One month: water arrives, some is released, the rest carries.

        Both state variables appear twice. `state.storage` is the level
        left by last month and `model.storage` the level left for next
        month; `state.inflow` is last month's inflow and `model.inflow`
        this month's, which is what makes the lag available.
        """
        model.storage = pyo.Var(bounds=(0, CAPACITY))
        model.inflow = pyo.Var(bounds=(0, CAPACITY))
        model.generate = pyo.Var(bounds=(0, TURBINE))
        model.spill = pyo.Var(bounds=(0, CAPACITY))

        # (3) the autoregressive step, affine in the carried inflow
        model.autoregressive = pyo.Constraint(
            expr=model.inflow == RHO * state.inflow + node.get("noise", 0.0)
        )
        model.balance = pyo.Constraint(
            expr=model.storage
            == state.storage + model.inflow - model.generate - model.spill
        )

        # (1) the price this stage happens to fetch
        return PRICE[node.stage] * model.generate

    msp.set_initial_state(storage=INITIAL_STORAGE, inflow=INITIAL_INFLOW)

    # (2) a different distribution at every stage
    msp.set_stage_realizations([realizations_for(stage) for stage in sorted(SEASONS)])
    return msp


def report_path(result, label: str, choose) -> None:
    """Walk one branch of the lattice through the finished policy."""
    path = ["0-0"]
    for stage in sorted(SEASONS):
        path.append(f"{stage}-{choose(stage)}")

    print(f"\n{label}:")
    print(f"  {'stage':<8}{'price':>7}{'inflow':>9}{'release':>9}{'storage':>9}")
    for step in result.simulate(path):
        stage = int(step.node_idx.split("-")[0])
        # the last stage passes nothing on, so read its own variables
        storage, inflow = (
            step.next_state
            if step.next_state
            else (step.solution.get("storage"), step.solution.get("inflow"))
        )
        release = step.solution.get("generate", float("nan"))
        season = SEASONS[stage][0] if stage in SEASONS else "start"
        print(
            f"  {season:<8}{PRICE[stage]:>7.0f}{inflow:>9.1f}"
            f"{release:>9.1f}{storage:>9.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="appsi_highs")
    parser.add_argument(
        "--plot", action="store_true", help="write the convergence chart"
    )
    args = parser.parse_args()

    # pyodsp emits log records but installs no handler; opt in to see progress.
    pyodsp.configure_logging()

    msp = build(args.solver)
    print(msp.describe(), end="\n\n")

    result = msp.solve()
    print(result.summary())

    # The bound is a single number; what the policy actually earns is a
    # distribution over scenario paths, and this is the sample the
    # convergence interval summarizes.
    print("\nsimulated objective over the sampled paths:")
    for name, value in result.objective_stats().items():
        print(f"  {name:<18}{value:>12,.2f}")

    # The wettest branch of every season, then the driest. The inflow
    # column shows the autocorrelation at work: each figure is 0.6 of the
    # one above it plus that season's noise.
    report_path(result, "wettest outcome each season", lambda stage: 0)
    report_path(
        result,
        "driest outcome each season",
        lambda stage: len(SEASONS[stage][1]) - 1,
    )

    if args.plot:
        # A policy has no trajectory of its own — it has one per scenario
        # — so the charts are drawn along the two branches reported above.
        paths = {
            "wettest": ["0-0"] + [f"{s}-0" for s in sorted(SEASONS)],
            "driest": ["0-0"]
            + [f"{s}-{len(SEASONS[s][1]) - 1}" for s in sorted(SEASONS)],
        }
        print()
        for written in result.plot(paths):
            print(f"wrote {written}")


if __name__ == "__main__":
    main()

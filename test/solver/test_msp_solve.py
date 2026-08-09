"""End-to-end multistage runs, against answers computed independently."""

import logging

import pyomo.environ as pyo
import pytest

from pyodsp.model.msp import MultistageProgram
from pyodsp.model.sp import StochasticProgram

SOLVER = "appsi_highs"


def inventory_program(tmp_path, *, stages=3, later_cost=5.0, realizations=None, **kw):
    """Buy now at 2 or later at `later_cost`, against a demand each stage."""
    kw.setdefault("log_level", logging.CRITICAL)
    msp = MultistageProgram(
        "inv",
        sense="min",
        solver=SOLVER,
        stage_bound=0.0,
        output_dir=tmp_path,
        sample_frequency=5,
        sample_size=10,
        **kw,
    )

    @msp.stage(state=["inventory"])
    def stage(m, incoming, node):
        m.inventory = pyo.Var(bounds=(0, 100))
        m.buy = pyo.Var(bounds=(0, 100))
        m.balance = pyo.Constraint(
            expr=m.inventory == incoming.inventory + m.buy - node.get("demand", 0.0)
        )
        return (2.0 if node.is_first else later_cost) * m.buy

    msp.set_initial_state(inventory=0.0)
    msp.set_realizations(
        realizations or [{"name": "d", "probability": 1.0, "demand": 10.0}],
        stages=stages,
    )
    return msp


# -- against hand computation ----------------------------------------------


def test_it_stocks_up_when_buying_later_is_dearer(tmp_path):
    # Stages 1 and 2 each need 10. Buying all 20 up front costs 2*20 = 40;
    # buying as needed costs 5*20 = 100.
    result = inventory_program(tmp_path).solve()

    assert result.bound == pytest.approx(40.0, abs=1e-6)
    assert result.first_stage_flat["inventory"] == pytest.approx(20.0, abs=1e-6)


def test_it_buys_as_needed_when_waiting_is_cheaper(tmp_path):
    # Buying later at 1 beats buying now at 2, so nothing is stocked up.
    result = inventory_program(tmp_path, later_cost=1.0).solve()

    assert result.bound == pytest.approx(20.0, abs=1e-6)
    assert result.first_stage_flat["inventory"] == pytest.approx(0.0, abs=1e-6)


def test_the_state_carries_through_every_stage(tmp_path):
    result = inventory_program(tmp_path, stages=4).solve()

    # three stages of demand now, still cheapest bought up front
    assert result.bound == pytest.approx(60.0, abs=1e-6)
    assert result.first_stage_flat["inventory"] == pytest.approx(30.0, abs=1e-6)


# -- against the two-stage pipeline ----------------------------------------


def two_stage_pair(tmp_path):
    """The same capacity problem stated both ways."""
    demand = {"lo": 3.0, "mid": 7.0, "hi": 11.0}
    prob = {"lo": 0.25, "mid": 0.5, "hi": 0.25}
    realizations = [
        {"name": n, "probability": prob[n], "demand": d} for n, d in demand.items()
    ]

    sp = StochasticProgram(
        "cap",
        sense="min",
        solver=SOLVER,
        output_dir=tmp_path / "sp",
        log_level=logging.CRITICAL,
    )

    @sp.first_stage
    def first_stage(m):
        m.x = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeReals)
        return 2.0 * m.x

    @sp.recourse
    def recourse(m, state, scenario):
        m.short = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeReals)
        m.meet = pyo.Constraint(expr=state.x + m.short >= scenario["demand"])
        return 7.0 * m.short

    sp.set_scenarios(realizations)

    msp = MultistageProgram(
        "cap",
        sense="min",
        solver=SOLVER,
        stage_bound=0.0,
        output_dir=tmp_path / "msp",
        log_level=logging.CRITICAL,
        sample_frequency=5,
        sample_size=20,
    )

    @msp.stage(state=["x"])
    def stage(m, incoming, node):
        m.x = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeReals)
        if node.is_first:
            return 2.0 * m.x
        m.short = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeReals)
        m.meet = pyo.Constraint(expr=incoming.x + m.short >= node["demand"])
        return 7.0 * m.short

    msp.set_initial_state(x=0.0)
    msp.set_realizations(realizations, stages=2)
    return sp, msp


def test_a_two_stage_lattice_reproduces_the_two_stage_pipeline(tmp_path):
    """The strongest available cross-check: two independent code paths.

    A multistage program with two stages *is* a two-stage program, so
    SDDP over the lattice and Benders over the tree must land on the same
    optimum and the same here-and-now decision.
    """
    sp, msp = two_stage_pair(tmp_path)

    sp_result = sp.solve()
    msp_result = msp.solve()

    assert msp_result.bound == pytest.approx(sp_result.objective, abs=1e-6)
    assert msp_result.first_stage_flat["x"] == pytest.approx(
        sp_result.first_stage_flat["x"], abs=1e-4
    )


# -- genuinely stochastic ---------------------------------------------------


def test_a_stochastic_lattice_lands_between_its_extremes(tmp_path):
    """Bracketed by the two deterministic problems it sits between.

    With demand 10 or 30 each stage, the optimum cannot beat always-10 nor
    cost more than always-30.
    """
    realizations = [
        {"name": "low", "probability": 0.5, "demand": 10.0},
        {"name": "high", "probability": 0.5, "demand": 30.0},
    ]
    stochastic = inventory_program(tmp_path / "s", realizations=realizations).solve()
    low = inventory_program(
        tmp_path / "l",
        realizations=[{"name": "d", "probability": 1.0, "demand": 10.0}],
    ).solve()
    high = inventory_program(
        tmp_path / "h",
        realizations=[{"name": "d", "probability": 1.0, "demand": 30.0}],
    ).solve()

    assert low.bound <= stochastic.bound + 1e-6
    assert stochastic.bound <= high.bound + 1e-6


def test_markov_transitions_are_honoured(tmp_path):
    """An absorbing chain started in the low state never sees the high one.

    So it has to cost exactly what the always-low problem costs.
    """
    realizations = [
        {"name": "low", "probability": 0.5, "demand": 10.0},
        {"name": "high", "probability": 0.5, "demand": 30.0},
    ]
    msp = inventory_program(tmp_path / "m", realizations=realizations)
    msp.set_markov_realizations(
        realizations,
        stages=3,
        transition_matrix=[[1.0, 0.0], [0.0, 1.0]],
        initial_distribution=[1.0, 0.0],
    )

    result = msp.solve()
    always_low = inventory_program(
        tmp_path / "l",
        realizations=[{"name": "d", "probability": 1.0, "demand": 10.0}],
    ).solve()

    assert result.bound == pytest.approx(always_low.bound, abs=1e-6)


# -- maximize ---------------------------------------------------------------


def test_a_maximize_program_comes_back_in_its_own_units(tmp_path):
    """Every coefficient negated, so the same decisions are optimal."""
    kw = dict(log_level=logging.CRITICAL)
    msp = MultistageProgram(
        "inv",
        sense="max",
        solver=SOLVER,
        stage_bound=0.0,
        output_dir=tmp_path,
        sample_frequency=5,
        sample_size=10,
        **kw,
    )

    @msp.stage(state=["inventory"])
    def stage(m, incoming, node):
        m.inventory = pyo.Var(bounds=(0, 100))
        m.buy = pyo.Var(bounds=(0, 100))
        m.balance = pyo.Constraint(
            expr=m.inventory == incoming.inventory + m.buy - node.get("demand", 0.0)
        )
        return -(2.0 if node.is_first else 5.0) * m.buy

    msp.set_initial_state(inventory=0.0)
    msp.set_realizations([{"name": "d", "probability": 1.0, "demand": 10.0}], stages=3)

    result = msp.solve()

    assert result.bound == pytest.approx(-40.0, abs=1e-6)
    assert result.first_stage_flat["inventory"] == pytest.approx(20.0, abs=1e-6)


# -- the policy -------------------------------------------------------------


def test_the_policy_replays_a_scenario_path(tmp_path):
    """SDDP's answer is a decision rule, not a schedule.

    Walking a path stage by stage against the cuts has to reproduce the
    same total the bound reports.
    """
    result = inventory_program(tmp_path).solve()

    trajectory = result.simulate(["0-0", "1-0", "2-0"])

    assert [s.node_idx for s in trajectory] == ["0-0", "1-0", "2-0"]
    assert trajectory[0].next_state[0] == pytest.approx(20.0, abs=1e-6)
    assert trajectory[1].next_state[0] == pytest.approx(10.0, abs=1e-6)
    assert trajectory[2].next_state == []
    assert sum(s.stage_cost for s in trajectory) == pytest.approx(40.0, abs=1e-6)


# -- reporting --------------------------------------------------------------


def test_the_result_reports_the_lattice_and_a_history(tmp_path):
    result = inventory_program(tmp_path, stages=4).solve()

    assert result.num_stages == 4
    assert result.nodes_per_stage == [1, 1, 1, 1]
    assert len(result.history) > 0
    assert list(result.history.columns) == ["iteration", "bound", "incumbent"]
    assert "via sddp" in result.summary()


def test_the_convergence_chart_is_written(tmp_path):
    pytest.importorskip("matplotlib")

    result = inventory_program(tmp_path).solve()
    path = result.plot_convergence()

    assert path.exists() and path.stat().st_size > 0


# -- time-dependent uncertainty --------------------------------------------


def test_a_distribution_that_changes_with_the_stage(tmp_path):
    """Stage 2 is certain to demand 30, so the whole path is determined.

    Stage 1 demands 10 either way, so the optimum is buy-everything-early:
    (10 + 30) * 2 = 80.
    """
    msp = MultistageProgram(
        "tv",
        sense="min",
        solver=SOLVER,
        stage_bound=0.0,
        output_dir=tmp_path,
        log_level=logging.CRITICAL,
        sample_frequency=5,
        sample_size=10,
    )

    @msp.stage(state=["inventory"])
    def stage(m, incoming, node):
        m.inventory = pyo.Var(bounds=(0, 200))
        m.buy = pyo.Var(bounds=(0, 200))
        m.balance = pyo.Constraint(
            expr=m.inventory == incoming.inventory + m.buy - node["demand"]
        )
        return (2.0 if node.is_first else 9.0) * m.buy

    msp.set_initial_state(inventory=0.0)
    msp.set_stage_realizations(
        [
            [{"name": "only", "probability": 1.0, "demand": 10.0}],
            [{"name": "only", "probability": 1.0, "demand": 30.0}],
        ],
        first_stage_data={"demand": 0.0},
    )

    result = msp.solve()

    assert result.bound == pytest.approx(80.0, abs=1e-6)
    assert result.first_stage_flat["inventory"] == pytest.approx(40.0, abs=1e-6)


def test_a_lagged_process_carried_as_state(tmp_path):
    """AR(1) inflows, handled by augmenting the state with the lag.

    SDDP needs the uncertainty to be stage-wise independent or Markov, so
    a process that depends on its own past is modelled by carrying that
    past in the state. With rho = 0.5, noise 10 and an initial inflow of
    80, the realized inflows are 40, then 30 — which is what the policy
    has to reproduce.
    """
    msp = MultistageProgram(
        "ar",
        sense="max",
        solver=SOLVER,
        stage_bound=0.0,
        output_dir=tmp_path,
        log_level=logging.CRITICAL,
        sample_frequency=5,
        sample_size=10,
    )
    rho = 0.5

    @msp.stage(state=["storage", "inflow"])
    def stage(m, incoming, node):
        m.storage = pyo.Var(bounds=(0, 500))
        m.inflow = pyo.Var(bounds=(0, 500))
        m.release = pyo.Var(bounds=(0, 30))
        m.spill = pyo.Var(bounds=(0, 500))
        m.ar = pyo.Constraint(
            expr=m.inflow == rho * incoming.inflow + node.get("noise", 0.0)
        )
        m.balance = pyo.Constraint(
            expr=m.storage == incoming.storage + m.inflow - m.release - m.spill
        )
        return 1.0 * m.release

    msp.set_initial_state(storage=0.0, inflow=80.0)
    msp.set_realizations([{"name": "n", "probability": 1.0, "noise": 10.0}], stages=3)

    trajectory = msp.solve().simulate(["0-0", "1-0", "2-0"])

    # next_state is (storage, inflow) in declaration order
    assert trajectory[0].next_state[1] == pytest.approx(40.0, abs=1e-6)
    assert trajectory[1].next_state[1] == pytest.approx(30.0, abs=1e-6)


def test_a_time_varying_transition_matrix(tmp_path):
    """Where you can go changes with the stage, not just with where you are.

    Stage 1 is forced into the low state and stage 2 into the high one, so
    the path is determined and costs 2*(10 + 30) = 80.
    """
    low_high = [
        {"name": "low", "probability": 0.5, "demand": 10.0},
        {"name": "high", "probability": 0.5, "demand": 30.0},
    ]
    msp = MultistageProgram(
        "tvt",
        sense="min",
        solver=SOLVER,
        stage_bound=0.0,
        output_dir=tmp_path,
        log_level=logging.CRITICAL,
        sample_frequency=5,
        sample_size=10,
    )

    @msp.stage(state=["inventory"])
    def stage(m, incoming, node):
        m.inventory = pyo.Var(bounds=(0, 200))
        m.buy = pyo.Var(bounds=(0, 200))
        m.balance = pyo.Constraint(
            expr=m.inventory == incoming.inventory + m.buy - node.get("demand", 0.0)
        )
        return (2.0 if node.is_first else 9.0) * m.buy

    msp.set_initial_state(inventory=0.0)
    msp.set_stage_realizations(
        [low_high, low_high],
        transitions=[
            [[1.0, 0.0]],  # stage 0 -> always low
            [[0.0, 1.0], [0.0, 1.0]],  # stage 1 -> always high
        ],
    )

    result = msp.solve()

    assert result.bound == pytest.approx(80.0, abs=1e-6)


# -- the simulated interval -------------------------------------------------


def test_the_run_records_the_intervals_it_tested_convergence_with(tmp_path):
    """SDDP's other side is estimated, so it comes with an interval.

    The bound is deterministic; the policy's value is a sample mean, and
    what stands against the bound is its confidence interval — recorded
    rather than only logged, so it can be reported and drawn.
    """
    result = inventory_program(tmp_path).solve()

    assert result.simulation is not None
    assert not result.simulation.empty
    assert list(result.simulation.columns) == [
        "iteration",
        "bound",
        "mean",
        "lower",
        "upper",
        "sample_size",
        "confidence_level",
    ]

    row = result.simulation.iloc[-1]
    assert row["lower"] <= row["mean"] <= row["upper"]
    assert row["sample_size"] == 10


def test_the_interval_brackets_the_bound_from_the_right_side(tmp_path):
    # Minimizing, the deterministic bound sits below the simulated policy.
    result = inventory_program(tmp_path).solve()

    row = result.simulation.iloc[-1]
    assert row["bound"] <= row["upper"] + 1e-6


def test_the_interval_is_reported_in_the_users_units(tmp_path):
    """A maximize run's bound sits *above* the simulated policy.

    Reported in the internal minimize convention the signs would flip and
    the ordering would read backwards.
    """
    msp = MultistageProgram(
        "inv",
        sense="max",
        solver=SOLVER,
        stage_bound=0.0,
        output_dir=tmp_path,
        log_level=logging.CRITICAL,
        sample_frequency=5,
        sample_size=10,
    )

    @msp.stage(state=["inventory"])
    def stage(m, incoming, node):
        m.inventory = pyo.Var(bounds=(0, 100))
        m.buy = pyo.Var(bounds=(0, 100))
        m.balance = pyo.Constraint(
            expr=m.inventory == incoming.inventory + m.buy - node.get("demand", 0.0)
        )
        return -(2.0 if node.is_first else 5.0) * m.buy

    msp.set_initial_state(inventory=0.0)
    msp.set_realizations([{"name": "d", "probability": 1.0, "demand": 10.0}], stages=3)

    result = msp.solve()
    row = result.simulation.iloc[-1]

    assert row["bound"] == pytest.approx(-40.0, abs=1e-6)
    assert row["lower"] <= row["mean"] <= row["upper"]
    assert row["bound"] >= row["lower"] - 1e-6


# -- the objective distribution --------------------------------------------


def test_the_run_keeps_the_draws_behind_its_last_interval(tmp_path):
    """The interval is a summary; the sample is what it summarizes.

    Whether the outcomes behind an expectation are tight or spread, and
    whether they are skewed, is not recoverable from a mean and two
    limits.
    """
    realizations = [
        {"name": "low", "probability": 0.5, "demand": 10.0},
        {"name": "high", "probability": 0.5, "demand": 30.0},
    ]
    result = inventory_program(tmp_path, realizations=realizations).solve()

    assert result.simulation_samples is not None
    assert len(result.simulation_samples) == 10
    # a genuinely stochastic problem gives genuinely different paths
    assert len(set(result.simulation_samples)) > 1


def test_the_stats_describe_the_sample_and_place_the_bound_beside_it(tmp_path):
    realizations = [
        {"name": "low", "probability": 0.5, "demand": 10.0},
        {"name": "high", "probability": 0.5, "demand": 30.0},
    ]
    result = inventory_program(tmp_path, realizations=realizations).solve()

    stats = result.objective_stats()

    assert stats["count"] == 10
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert stats["ci_lower"] <= stats["mean"] <= stats["ci_upper"]
    assert stats["bound"] == pytest.approx(result.bound)
    assert stats["confidence_level"] == pytest.approx(0.95)


def test_the_samples_are_in_the_users_units(tmp_path):
    """A maximize run's simulated objectives are positive here.

    Left in the internal minimize convention they would all be negative,
    and the histogram would sit on the wrong side of the bound.
    """
    msp = MultistageProgram(
        "inv",
        sense="max",
        solver=SOLVER,
        stage_bound=0.0,
        output_dir=tmp_path,
        log_level=logging.CRITICAL,
        sample_frequency=5,
        sample_size=10,
    )

    @msp.stage(state=["inventory"])
    def stage(m, incoming, node):
        m.inventory = pyo.Var(bounds=(0, 100))
        m.buy = pyo.Var(bounds=(0, 100))
        m.balance = pyo.Constraint(
            expr=m.inventory == incoming.inventory + m.buy - node.get("demand", 0.0)
        )
        return -(2.0 if node.is_first else 5.0) * m.buy

    msp.set_initial_state(inventory=0.0)
    msp.set_realizations([{"name": "d", "probability": 1.0, "demand": 10.0}], stages=3)

    result = msp.solve()

    assert all(value == pytest.approx(-40.0) for value in result.simulation_samples)


def test_the_distribution_chart_is_written(tmp_path):
    pytest.importorskip("matplotlib")

    realizations = [
        {"name": "low", "probability": 0.5, "demand": 10.0},
        {"name": "high", "probability": 0.5, "demand": 30.0},
    ]
    result = inventory_program(tmp_path, realizations=realizations).solve()

    written = result.plot_objective_distribution()

    assert written.exists() and written.stat().st_size > 0


def test_asking_for_a_distribution_a_run_never_produced(tmp_path):
    # A run that stopped before its first convergence test has no sample.
    result = inventory_program(tmp_path).solve()
    result.simulation_samples = None

    with pytest.raises(ValueError, match="no simulation samples"):
        result.objective_stats()
    with pytest.raises(ValueError, match="no simulation samples"):
        result.plot_objective_distribution()

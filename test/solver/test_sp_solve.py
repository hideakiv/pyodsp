"""End-to-end runs of the two-stage pipeline.

Every optimum here is checked against the extensive form solved directly,
so the tests fail if the decomposition returns a different answer from
the model it claims to decompose.
"""

import logging
import warnings

import pyomo.environ as pyo
import pytest

from pyodsp.model.sp import StochasticProgram

SOLVER = "appsi_highs"

DEMAND = {"lo": 3.0, "mid": 7.0, "hi": 11.0}
PROB = {"lo": 0.25, "mid": 0.5, "hi": 0.25}
BUILD_COST = 2.0
SHORT_COST = 7.0


# -- the reference model ----------------------------------------------------


def extensive_form(*, integer: bool = False, hold: float = 0.0):
    """The same program as one monolithic Pyomo model."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeReals)
    m.s = pyo.Set(initialize=list(DEMAND))
    m.short = pyo.Var(
        m.s,
        bounds=(0, 20),
        domain=pyo.NonNegativeIntegers if integer else pyo.NonNegativeReals,
    )
    m.meet = pyo.Constraint(m.s, rule=lambda m, s: m.x + m.short[s] >= DEMAND[s])
    m.obj = pyo.Objective(
        expr=BUILD_COST * m.x
        + sum(PROB[s] * (SHORT_COST * m.short[s] + hold * m.x) for s in m.s),
        sense=pyo.minimize,
    )
    pyo.SolverFactory(SOLVER).solve(m)
    return pyo.value(m.obj), pyo.value(m.x)


def capacity_program(tmp_path, *, integer=False, hold=0.0, **kwargs):
    kwargs.setdefault("log_level", logging.CRITICAL)
    sp = StochasticProgram(
        "cap", sense="min", solver=SOLVER, output_dir=tmp_path, **kwargs
    )

    @sp.first_stage
    def first_stage(m):
        m.x = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeReals)
        return BUILD_COST * m.x

    @sp.recourse
    def recourse(m, state, scenario):
        m.short = pyo.Var(
            bounds=(0, 20),
            domain=pyo.NonNegativeIntegers if integer else pyo.NonNegativeReals,
        )
        m.meet = pyo.Constraint(expr=state.x + m.short >= scenario["demand"])
        return SHORT_COST * m.short + hold * state.x

    sp.set_scenarios(
        [{"name": n, "probability": PROB[n], "demand": d} for n, d in DEMAND.items()]
    )
    return sp


# -- agreement with the extensive form --------------------------------------


def test_benders_matches_the_extensive_form(tmp_path):
    expected_objective, expected_x = extensive_form()

    result = capacity_program(tmp_path).solve()

    assert result.method == "bd"
    assert result.objective == pytest.approx(expected_objective, abs=1e-6)
    assert result.first_stage_flat["x"] == pytest.approx(expected_x, abs=1e-6)


def test_dual_decomposition_matches_the_extensive_form(tmp_path):
    expected_objective, expected_x = extensive_form()

    result = capacity_program(tmp_path, method="dd").solve()

    assert result.method == "dd"
    assert result.objective == pytest.approx(expected_objective, abs=1e-6)
    assert result.first_stage_flat["x"] == pytest.approx(expected_x, abs=1e-6)
    assert result.first_stage_consistent


def test_an_integer_recourse_is_solved_exactly_by_scaled_cuts(tmp_path):
    expected_objective, expected_x = extensive_form(integer=True)

    with pytest.warns(UserWarning, match="Switching to Benders with scaled cuts"):
        sp = capacity_program(tmp_path, integer=True)
        result = sp.solve()

    assert result.method == "bdsc"
    assert result.objective == pytest.approx(expected_objective, abs=1e-4)
    assert result.first_stage_flat["x"] == pytest.approx(expected_x, abs=1e-4)


def test_relaxing_the_recourse_gives_the_relaxation(tmp_path):
    relaxed_objective, relaxed_x = extensive_form(integer=False)

    with pytest.warns(UserWarning, match="Relaxing the integrality"):
        sp = capacity_program(tmp_path, integer=True, integer_recourse="relax")
        result = sp.solve()

    assert result.method == "bd"
    assert result.objective == pytest.approx(relaxed_objective, abs=1e-6)
    assert result.first_stage_flat["x"] == pytest.approx(relaxed_x, abs=1e-6)


def test_a_state_variable_in_the_recourse_objective_still_solves_correctly(tmp_path):
    """The regression that motivates state.add_state_link.

    Benders cuts are read off constraint duals, so a state variable used
    only in the recourse objective would contribute nothing to the cut
    gradient. Here that omission is not cosmetic: it moves the optimum
    from x = 3 to x = 7.
    """
    expected_objective, expected_x = extensive_form(hold=4.0)
    assert expected_x == pytest.approx(3.0, abs=1e-6)  # guard the guard

    result = capacity_program(tmp_path, hold=4.0).solve()

    assert result.objective == pytest.approx(expected_objective, abs=1e-6)
    assert result.first_stage_flat["x"] == pytest.approx(expected_x, abs=1e-6)


# -- the deterministic equivalent -------------------------------------------


def test_the_deterministic_equivalent_matches_the_reference_model(tmp_path):
    expected_objective, expected_x = extensive_form()

    result = capacity_program(tmp_path, method="de").solve()

    assert result.method == "de"
    assert result.objective == pytest.approx(expected_objective, abs=1e-6)
    assert result.first_stage_flat["x"] == pytest.approx(expected_x, abs=1e-6)


def test_it_solves_integrality_directly(tmp_path):
    # No LP duals are involved, so an integer recourse needs no special
    # handling and no warning — the solver is handed the MIP whole.
    expected_objective, expected_x = extensive_form(integer=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = capacity_program(tmp_path, method="de", integer=True).solve()

    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    assert result.objective == pytest.approx(expected_objective, abs=1e-6)
    assert result.first_stage_flat["x"] == pytest.approx(expected_x, abs=1e-6)


@pytest.mark.parametrize("method", ["bd", "dd"])
def test_every_decomposition_agrees_with_the_deterministic_equivalent(tmp_path, method):
    """What the method is for: the answer no decomposition should differ from."""
    reference = capacity_program(tmp_path / "de", method="de").solve()
    decomposed = capacity_program(tmp_path / method, method=method).solve()

    assert decomposed.objective == pytest.approx(reference.objective, abs=1e-6)
    assert decomposed.first_stage_flat["x"] == pytest.approx(
        reference.first_stage_flat["x"], abs=1e-6
    )


def test_it_reports_no_iterations_and_a_closed_gap(tmp_path):
    # One solve, proved optimal — there is no trajectory to record.
    result = capacity_program(tmp_path, method="de").solve()

    assert result.history.empty
    assert result.bound == result.objective
    assert result.gap == pytest.approx(0.0)


def test_it_reports_each_scenario_separately(tmp_path):
    result = capacity_program(tmp_path, method="de").solve()

    by_name = {s.name: s for s in result.scenarios}
    # with capacity x = 7 only the high-demand scenario falls short
    assert by_name["hi"].variables["short"] == pytest.approx(4.0, abs=1e-6)
    assert by_name["lo"].variables["short"] == pytest.approx(0.0, abs=1e-6)
    # each scenario's own recourse cost, unweighted
    assert by_name["hi"].objective == pytest.approx(SHORT_COST * 4.0, abs=1e-6)


def test_it_writes_its_solution(tmp_path):
    capacity_program(tmp_path, method="de").solve()

    assert (tmp_path / "sol.csv").exists()


def test_a_maximize_deterministic_equivalent_comes_back_in_its_own_units(tmp_path):
    result = farmer_program(tmp_path, method="de").solve()

    assert result.objective == pytest.approx(108390.0, abs=1e-3)
    assert result.first_stage["acres"] == pytest.approx(
        {"WHEAT": 170.0, "CORN": 80.0, "BEETS": 250.0}, abs=1e-3
    )


# -- maximize ---------------------------------------------------------------


def farmer_program(tmp_path, **kwargs):
    """Birge and Louveaux's farmer, whose optimum is 108390."""
    crops = ["WHEAT", "CORN", "BEETS"]
    cost = {"WHEAT": 150.0, "CORN": 230.0, "BEETS": 260.0}
    price = {"WHEAT": 170.0, "CORN": 150.0, "BEETS": 36.0}
    over = {"WHEAT": 0.0, "CORN": 0.0, "BEETS": 10.0}
    quota = {"WHEAT": 100000.0, "CORN": 100000.0, "BEETS": 6000.0}
    feed = {"WHEAT": 200.0, "CORN": 240.0, "BEETS": 0.0}
    buy = {"WHEAT": 238.0, "CORN": 210.0, "BEETS": 100000.0}
    yields = {
        "GOOD": {"WHEAT": 3.0, "CORN": 3.6, "BEETS": 24.0},
        "AVERAGE": {"WHEAT": 2.5, "CORN": 3.0, "BEETS": 20.0},
        "POOR": {"WHEAT": 2.0, "CORN": 2.4, "BEETS": 16.0},
    }

    kwargs.setdefault("log_level", logging.CRITICAL)
    sp = StochasticProgram(
        "farmer", sense="max", solver=SOLVER, output_dir=tmp_path, **kwargs
    )

    @sp.first_stage
    def first_stage(m):
        m.acres = pyo.Var(crops, domain=pyo.NonNegativeReals)
        m.land = pyo.Constraint(expr=sum(m.acres[c] for c in crops) <= 500)
        return -sum(cost[c] * m.acres[c] for c in crops)

    @sp.recourse
    def recourse(m, state, scenario):
        y = scenario["yield"]
        m.sub = pyo.Var(crops, domain=pyo.NonNegativeReals)
        m.over = pyo.Var(crops, domain=pyo.NonNegativeReals)
        m.keep = pyo.Var(crops, domain=pyo.NonNegativeReals)
        m.buy = pyo.Var(crops, domain=pyo.NonNegativeReals)
        m.balance = pyo.Constraint(
            crops,
            rule=lambda m, c: m.sub[c] + m.over[c] + m.keep[c] == y[c] * state.acres[c],
        )
        m.feed = pyo.Constraint(
            crops, rule=lambda m, c: m.keep[c] + m.buy[c] >= feed[c]
        )
        m.quota = pyo.Constraint(crops, rule=lambda m, c: m.sub[c] <= quota[c])
        return (
            sum(price[c] * m.sub[c] for c in crops)
            + sum(over[c] * m.over[c] for c in crops)
            - sum(buy[c] * m.buy[c] for c in crops)
        )

    sp.set_scenarios({name: {"yield": y} for name, y in yields.items()})
    return sp


def test_a_maximize_program_comes_back_in_its_own_units(tmp_path):
    result = farmer_program(tmp_path).solve()

    assert result.objective == pytest.approx(108390.0, abs=1e-3)
    assert result.bound == pytest.approx(108390.0, abs=1e-3)
    assert result.first_stage["acres"] == pytest.approx(
        {"WHEAT": 170.0, "CORN": 80.0, "BEETS": 250.0}, abs=1e-3
    )
    # a maximize run climbs toward the optimum rather than descending to it
    assert result.history["incumbent"].dropna().iloc[-1] == pytest.approx(
        108390.0, abs=1e-3
    )


def test_the_saved_trajectory_agrees_with_the_returned_result(tmp_path):
    import pandas as pd

    result = farmer_program(tmp_path).solve()
    saved = pd.read_csv(tmp_path / "node0" / "bm.csv", index_col=0)

    # the files on disk are corrected to the user's units too, not left
    # in the internal minimize convention
    assert saved["obj_bound"].iloc[-1] == pytest.approx(108390.0, abs=1e-3)
    assert saved["obj_bound"].iloc[-1] == pytest.approx(result.bound, abs=1e-9)


# -- result surface ---------------------------------------------------------


def test_the_result_reports_a_closed_gap_and_a_history(tmp_path):
    result = capacity_program(tmp_path).solve()

    assert result.gap == pytest.approx(0.0, abs=1e-6)
    assert len(result.history) > 1
    assert list(result.history.columns) == ["iteration", "bound", "incumbent"]
    assert len(result.scenarios) == 3
    assert sum(s.probability for s in result.scenarios) == pytest.approx(1.0)


def test_scenario_outcomes_carry_their_own_variable_values(tmp_path):
    result = capacity_program(tmp_path).solve()

    by_name = {s.name: s for s in result.scenarios}
    # with capacity x = 7 only the high-demand scenario falls short
    assert by_name["hi"].variables["short"] == pytest.approx(4.0, abs=1e-6)
    assert by_name["lo"].variables["short"] == pytest.approx(0.0, abs=1e-6)


def test_summary_mentions_the_algorithm_and_the_decision(tmp_path):
    result = capacity_program(tmp_path).solve()
    text = result.summary()

    assert "via bd" in text
    assert "x = 7" in text


def test_the_frames_line_up_with_the_result(tmp_path):
    result = capacity_program(tmp_path).solve()

    assert list(result.first_stage_frame()["variable"]) == ["x"]
    assert list(result.scenario_frame()["scenario"]) == ["lo", "mid", "hi"]


# -- plotting ---------------------------------------------------------------


def test_plotting_writes_every_applicable_chart(tmp_path):
    pytest.importorskip("matplotlib")

    result = capacity_program(tmp_path).solve()
    written = result.plot(tmp_path / "charts")

    assert {p.name for p in written} == {
        "convergence.png",
        "scenario_objectives.png",
        "scenario_tree.png",
    }
    assert all(p.stat().st_size > 0 for p in written)

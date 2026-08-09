"""EVPI and VSS, against a hand-computed instance.

Capacity x costs 2 per unit; a shortfall costs 7. Demand is 3, 7 or 11
with probabilities 0.5, 0.2, 0.3 — skewed on purpose, so that planning
against the mean gives a different decision from the stochastic program
and VSS is not trivially zero.

    mean demand = 0.5(3) + 0.2(7) + 0.3(11)        = 6.2
    WS          = 2 * mean demand                   = 12.4
    RP          : x = 11 (P(d > 7) = 0.3 > 2/7)     = 22.0
    EEV         : x = 6.2, plus expected shortfall  = 23.6
    EVPI        = 22.0 - 12.4                       =  9.6
    VSS         = 23.6 - 22.0                       =  1.6
"""

import logging

import pyomo.environ as pyo
import pytest

from pyodsp.model.sp import StochasticProgram

SOLVER = "appsi_highs"
DEMAND = {"lo": 3.0, "mid": 7.0, "hi": 11.0}
PROB = {"lo": 0.5, "mid": 0.2, "hi": 0.3}
BUILD_COST = 2.0
SHORT_COST = 7.0

WS = 12.4
RP = 22.0
EEV = 23.6
EVPI = 9.6
VSS = 1.6


def capacity_program(tmp_path, *, sense="min", **kwargs):
    """The instance above. Maximizing negates every coefficient, so the
    same decisions are optimal and every value negates with them."""
    flip = -1.0 if sense == "max" else 1.0
    kwargs.setdefault("log_level", logging.CRITICAL)
    sp = StochasticProgram(
        "cap", sense=sense, solver=SOLVER, output_dir=tmp_path, **kwargs
    )

    @sp.first_stage
    def first_stage(m):
        m.x = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeReals)
        return flip * BUILD_COST * m.x

    @sp.recourse
    def recourse(m, state, scenario):
        m.short = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeReals)
        m.meet = pyo.Constraint(expr=state.x + m.short >= scenario["demand"])
        return flip * SHORT_COST * m.short

    sp.set_scenarios(
        [{"name": n, "probability": PROB[n], "demand": d} for n, d in DEMAND.items()]
    )
    return sp


# -- the measures -----------------------------------------------------------


def test_the_measures_match_the_hand_computed_instance(tmp_path):
    analysis = capacity_program(tmp_path).analyze()

    assert analysis.ws == pytest.approx(WS, abs=1e-6)
    assert analysis.rp.objective == pytest.approx(RP, abs=1e-6)
    assert analysis.eev == pytest.approx(EEV, abs=1e-6)
    assert analysis.evpi == pytest.approx(EVPI, abs=1e-6)
    assert analysis.vss == pytest.approx(VSS, abs=1e-6)


def test_the_orderings_hold(tmp_path):
    # For a minimization, perfect information is cheapest and the
    # mean-value decision dearest.
    analysis = capacity_program(tmp_path).analyze()

    assert analysis.ws <= analysis.rp.objective <= analysis.eev


def test_both_measures_stay_positive_when_maximizing(tmp_path):
    """The orderings reverse, so the differences have to be signed.

    Reported unsigned, EVPI and VSS would come out negative here — the
    same information being worth the same amount.
    """
    analysis = capacity_program(tmp_path, sense="max").analyze()

    assert analysis.rp.objective == pytest.approx(-RP, abs=1e-6)
    assert analysis.ws == pytest.approx(-WS, abs=1e-6)
    assert analysis.eev == pytest.approx(-EEV, abs=1e-6)
    assert analysis.evpi == pytest.approx(EVPI, abs=1e-6)
    assert analysis.vss == pytest.approx(VSS, abs=1e-6)
    assert analysis.ws >= analysis.rp.objective >= analysis.eev


def test_the_expected_value_decision_is_the_mean_demand(tmp_path):
    analysis = capacity_program(tmp_path).analyze()

    assert analysis.ev.first_stage_flat["x"] == pytest.approx(6.2, abs=1e-6)
    # the stochastic program builds more, which is what VSS measures
    assert analysis.rp.first_stage_flat["x"] == pytest.approx(11.0, abs=1e-6)


def test_wait_and_see_keeps_each_scenarios_own_optimum(tmp_path):
    analysis = capacity_program(tmp_path).analyze()

    # knowing the demand, you build exactly it and never fall short
    assert analysis.scenario_values["hi"] == pytest.approx(2.0 * 11.0, abs=1e-6)
    assert analysis.scenario_values["lo"] == pytest.approx(2.0 * 3.0, abs=1e-6)


def test_relative_measures_are_fractions_of_rp(tmp_path):
    analysis = capacity_program(tmp_path).analyze()

    assert analysis.relative_vss == pytest.approx(VSS / RP, abs=1e-9)
    assert analysis.relative_evpi == pytest.approx(EVPI / RP, abs=1e-9)


# -- a degenerate case ------------------------------------------------------


def test_vss_is_zero_when_the_mean_decision_is_already_optimal(tmp_path):
    # Symmetric probabilities put the mean at the optimal quantile, so
    # planning against the mean loses nothing — VSS is zero, not negative.
    sp = capacity_program(tmp_path)
    sp.set_scenarios(
        [
            {"name": "lo", "probability": 0.25, "demand": 3.0},
            {"name": "mid", "probability": 0.5, "demand": 7.0},
            {"name": "hi", "probability": 0.25, "demand": 11.0},
        ]
    )

    analysis = sp.analyze()

    assert analysis.vss == pytest.approx(0.0, abs=1e-6)
    assert analysis.evpi > 0.0


# -- selecting measures -----------------------------------------------------


def test_asking_for_one_measure_skips_the_others_solves(tmp_path):
    analysis = capacity_program(tmp_path).analyze(measures=("evpi",))

    assert analysis.evpi == pytest.approx(EVPI, abs=1e-6)
    assert analysis.ws is not None
    # the expected-value problem was never built
    assert analysis.ev is None
    assert analysis.eev is None
    assert analysis.vss is None


def test_vss_alone_skips_the_per_scenario_solves(tmp_path):
    analysis = capacity_program(tmp_path).analyze(measures=("vss",))

    assert analysis.vss == pytest.approx(VSS, abs=1e-6)
    assert analysis.ws is None
    assert analysis.evpi is None


@pytest.mark.parametrize("measures", [(), ("nonsense",)])
def test_unusable_measure_selections_are_refused(tmp_path, measures):
    with pytest.raises(ValueError):
        capacity_program(tmp_path).analyze(measures=measures)


# -- an EV decision that does not survive contact with the scenarios --------


def test_an_infeasible_mean_value_decision_is_reported_not_silently_dropped(tmp_path):
    """VSS is unbounded, which is a stronger statement than a large one."""
    sp = StochasticProgram(
        "tight",
        sense="min",
        solver=SOLVER,
        output_dir=tmp_path,
        log_level=logging.CRITICAL,
    )

    @sp.first_stage
    def first_stage(m):
        m.x = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeReals)
        return m.x

    @sp.recourse
    def recourse(m, state, scenario):
        # no recourse variable at all: the first stage must cover demand
        # by itself, and the mean decision cannot cover the high scenario
        m.meet = pyo.Constraint(expr=state.x >= scenario["demand"])
        return 0.0

    sp.set_scenarios(
        [{"name": n, "probability": PROB[n], "demand": d} for n, d in DEMAND.items()]
    )

    analysis = sp.analyze(measures=("vss",))

    assert analysis.eev_infeasible
    assert analysis.eev is None
    assert analysis.vss is None
    assert "infeasible" in analysis.summary()


# -- reporting --------------------------------------------------------------


def test_summary_names_the_three_values_and_both_gaps(tmp_path):
    text = capacity_program(tmp_path).analyze().summary()

    for token in ("WS", "RP", "EEV", "EVPI", "VSS", "% of RP"):
        assert token in text


def test_the_frame_carries_every_measure(tmp_path):
    frame = capacity_program(tmp_path).analyze().to_frame()

    assert list(frame["measure"]) == ["WS", "RP", "EEV", "EVPI", "VSS"]
    assert frame["value"].notna().all()


def test_plotting_writes_the_chart(tmp_path):
    pytest.importorskip("matplotlib")

    analysis = capacity_program(tmp_path).analyze()
    path = analysis.plot()

    assert path.exists() and path.stat().st_size > 0
    assert path.name == "information_value.png"

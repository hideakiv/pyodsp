"""CVaR end to end, against hand computation and across methods.

Capacity x costs 2 per unit; a shortfall costs 7. Demand is 3, 7 or 11
with probabilities 0.5, 0.3, 0.2 — chosen so P(demand > 7) = 0.2 < 2/7,
which means the risk-neutral optimum deliberately under-hedges at x = 7
and there is a tail for risk aversion to push against.

    risk-neutral : x = 7,  cost 14 + 7(0.2)(4)                    = 19.6
    CVaR a=0.8   : the tail is exactly the high scenario, so the
                   objective is 2x + 7(11-x)+, minimized at x = 11  = 22.0
"""

import logging

import pyomo.environ as pyo
import pytest

from pyodsp.model.sp import CVaR, StochasticProgram

SOLVER = "appsi_highs"
DEMAND = {"lo": 3.0, "mid": 7.0, "hi": 11.0}
PROB = {"lo": 0.5, "mid": 0.3, "hi": 0.2}

NEUTRAL_OBJECTIVE, NEUTRAL_X = 19.6, 7.0
AVERSE_OBJECTIVE, AVERSE_X = 22.0, 11.0


def capacity_program(tmp_path, *, method="bd", risk=None, sense="min", **kwargs):
    flip = -1.0 if sense == "max" else 1.0
    kwargs.setdefault("log_level", logging.CRITICAL)
    sp = StochasticProgram(
        "cap",
        sense=sense,
        method=method,
        risk=risk,
        solver=SOLVER,
        output_dir=tmp_path,
        **kwargs,
    )

    @sp.first_stage
    def first_stage(m):
        m.x = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeReals)
        return flip * 2.0 * m.x

    @sp.recourse
    def recourse(m, state, scenario):
        m.short = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeReals)
        m.meet = pyo.Constraint(expr=state.x + m.short >= scenario["demand"])
        return flip * 7.0 * m.short

    sp.set_scenarios(
        [{"name": n, "probability": PROB[n], "demand": d} for n, d in DEMAND.items()]
    )
    return sp


# -- against hand computation ----------------------------------------------


@pytest.mark.parametrize("method", ["de", "bd"])
def test_risk_aversion_buys_the_hedge_the_expectation_declines(tmp_path, method):
    neutral = capacity_program(tmp_path / "n", method=method).solve()
    averse = capacity_program(
        tmp_path / "a", method=method, risk=CVaR(alpha=0.8, weight=1.0)
    ).solve()

    assert neutral.objective == pytest.approx(NEUTRAL_OBJECTIVE, abs=1e-6)
    assert neutral.first_stage_flat["x"] == pytest.approx(NEUTRAL_X, abs=1e-6)

    assert averse.objective == pytest.approx(AVERSE_OBJECTIVE, abs=1e-6)
    assert averse.first_stage_flat["x"] == pytest.approx(AVERSE_X, abs=1e-6)


@pytest.mark.parametrize(
    "risk",
    [
        None,
        CVaR(alpha=0.8, weight=0.0),
        CVaR(alpha=0.8, weight=0.5),
        CVaR(alpha=0.8, weight=1.0),
        CVaR(alpha=0.5, weight=0.7),
    ],
    ids=["neutral", "weight0", "half", "full", "wider-tail"],
)
def test_benders_reproduces_the_deterministic_equivalent(tmp_path, risk):
    """Two independent formulations of the same risk-averse problem.

    The deterministic equivalent states Rockafellar-Uryasev directly on
    the scenario costs; Benders states it on the master's per-scenario
    thetas, which carry probabilities the other does not. They have to
    land on the same answer.
    """
    reference = capacity_program(tmp_path / "de", method="de", risk=risk).solve()
    decomposed = capacity_program(tmp_path / "bd", method="bd", risk=risk).solve()

    assert decomposed.objective == pytest.approx(reference.objective, abs=1e-6)
    assert decomposed.first_stage_flat["x"] == pytest.approx(
        reference.first_stage_flat["x"], abs=1e-4
    )


@pytest.mark.parametrize("method", ["de", "bd"])
def test_no_weight_on_the_tail_is_exactly_risk_neutral(tmp_path, method):
    """The free correctness check on the whole risk apparatus.

    weight=0 builds every extra variable and constraint CVaR needs and
    then gives them no say, so it must reproduce the risk-neutral answer
    exactly — not merely closely.
    """
    neutral = capacity_program(tmp_path / "n", method=method).solve()
    zero_weight = capacity_program(
        tmp_path / "z", method=method, risk=CVaR(alpha=0.9, weight=0.0)
    ).solve()

    assert zero_weight.objective == pytest.approx(neutral.objective, abs=1e-9)
    assert zero_weight.first_stage_flat["x"] == pytest.approx(
        neutral.first_stage_flat["x"], abs=1e-9
    )


def test_more_aversion_never_lowers_the_expected_cost(tmp_path):
    """What risk aversion costs, which is the point of reporting both.

    The risk-averse solution is optimal for the tail, so it cannot also
    beat the risk-neutral one on average.
    """
    neutral = capacity_program(tmp_path / "n").solve()
    averse = capacity_program(tmp_path / "a", risk=CVaR(alpha=0.8, weight=1.0)).solve()

    assert averse.expected_objective >= neutral.expected_objective - 1e-6


# -- sense ------------------------------------------------------------------


@pytest.mark.parametrize("method", ["de", "bd"])
def test_maximizing_attends_to_the_low_tail(tmp_path, method):
    """The mirrored problem must mirror exactly.

    Every coefficient is negated, so the same decisions are optimal and
    every value negates with them. The tail CVaR attends to is the upper
    one of the converted objective, which is the *lower* one of the
    original — the outcomes a maximizer fears.
    """
    averse_min = capacity_program(
        tmp_path / "min", method=method, risk=CVaR(alpha=0.8, weight=1.0)
    ).solve()
    averse_max = capacity_program(
        tmp_path / "max",
        method=method,
        sense="max",
        risk=CVaR(alpha=0.8, weight=1.0),
    ).solve()

    assert averse_max.objective == pytest.approx(-averse_min.objective, abs=1e-6)
    assert averse_max.first_stage_flat["x"] == pytest.approx(AVERSE_X, abs=1e-6)


# -- reporting --------------------------------------------------------------


def test_both_values_are_reported(tmp_path):
    result = capacity_program(tmp_path, risk=CVaR(alpha=0.8, weight=1.0)).solve()

    assert result.risk.describe() == "CVaR(alpha=0.8, weight=1)"
    assert result.objective == pytest.approx(AVERSE_OBJECTIVE, abs=1e-6)
    assert result.expected_objective is not None

    text = result.summary()
    assert "risk-adjusted" in text
    assert "expected" in text


def test_a_risk_neutral_run_reports_one_value(tmp_path):
    result = capacity_program(tmp_path).solve()

    assert result.objective == pytest.approx(result.expected_objective, abs=1e-9)
    assert "risk-adjusted" not in result.summary()


def test_describe_names_the_risk_measure(tmp_path):
    sp = capacity_program(tmp_path, risk=CVaR(alpha=0.9, weight=0.3))

    assert "CVaR(alpha=0.9, weight=0.3)" in sp.describe()

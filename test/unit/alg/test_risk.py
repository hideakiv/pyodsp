"""Risk measures, and the exact CVaR of a discrete distribution."""

import pytest

from pyodsp.alg.risk import CVaR, Expectation, cvar_of_sample, value_of_sample

# Costs in the minimize convention: larger is worse.
VALUES = [10.0, 20.0, 30.0]
PROBABILITIES = [0.5, 0.3, 0.2]
MEAN = 17.0  # 0.5*10 + 0.3*20 + 0.2*30


# -- the measures themselves ------------------------------------------------


def test_expectation_is_risk_neutral():
    assert Expectation().is_risk_neutral


@pytest.mark.parametrize(
    "risk", [CVaR(alpha=0.9, weight=0.0), CVaR(alpha=0.0, weight=1.0)]
)
def test_cvar_reduces_to_risk_neutral_at_its_edges(risk):
    # Either no weight on the tail, or a tail that is the whole
    # distribution. Both are the expectation, and worth being explicit
    # about: weight=0 is the check that CVaR machinery has not changed
    # the risk-neutral answer.
    assert risk.is_risk_neutral


def test_the_tail_is_what_is_left_above_alpha():
    assert CVaR(alpha=0.95).tail == pytest.approx(0.05)


@pytest.mark.parametrize("alpha", [-0.1, 1.0, 1.5])
def test_an_unusable_alpha_is_refused(alpha):
    with pytest.raises(ValueError, match="alpha must be"):
        CVaR(alpha=alpha)


@pytest.mark.parametrize("weight", [-0.1, 1.5])
def test_an_unusable_weight_is_refused(weight):
    with pytest.raises(ValueError, match="weight must be"):
        CVaR(weight=weight)


# -- the discrete CVaR ------------------------------------------------------


def test_a_tail_that_is_exactly_one_outcome():
    # tail 0.2 is precisely the worst scenario's probability
    assert cvar_of_sample(VALUES, PROBABILITIES, alpha=0.8) == pytest.approx(30.0)


def test_a_tail_that_straddles_two_outcomes():
    # tail 0.5 takes all of the worst (0.2 at 30) and part of the next
    # (0.3 at 20): (0.2*30 + 0.3*20) / 0.5
    assert cvar_of_sample(VALUES, PROBABILITIES, alpha=0.5) == pytest.approx(24.0)


def test_a_tail_smaller_than_the_worst_outcome():
    # entirely inside the worst scenario, so it is that scenario's value
    assert cvar_of_sample(VALUES, PROBABILITIES, alpha=0.95) == pytest.approx(30.0)


def test_the_whole_distribution_is_the_mean():
    assert cvar_of_sample(VALUES, PROBABILITIES, alpha=0.0) == pytest.approx(MEAN)


def test_cvar_is_never_below_the_mean():
    # It averages the worst outcomes, so it cannot be better than
    # averaging all of them.
    for alpha in (0.0, 0.25, 0.5, 0.75, 0.9):
        assert cvar_of_sample(VALUES, PROBABILITIES, alpha) >= MEAN - 1e-9


def test_ordering_of_the_outcomes_does_not_matter():
    shuffled = [30.0, 10.0, 20.0]
    weights = [0.2, 0.5, 0.3]

    assert cvar_of_sample(shuffled, weights, 0.5) == pytest.approx(24.0)


def test_a_degenerate_distribution_is_its_own_cvar():
    assert cvar_of_sample([5.0, 5.0], [0.5, 0.5], 0.9) == pytest.approx(5.0)


def test_no_outcomes_is_refused():
    with pytest.raises(ValueError, match="No outcomes"):
        cvar_of_sample([], [], 0.9)


# -- blending ---------------------------------------------------------------


def test_the_expectation_values_a_sample_by_its_mean():
    assert value_of_sample(Expectation(), VALUES, PROBABILITIES) == pytest.approx(MEAN)


def test_a_blend_sits_between_the_mean_and_the_tail():
    tail = cvar_of_sample(VALUES, PROBABILITIES, 0.8)
    blended = value_of_sample(CVaR(alpha=0.8, weight=0.25), VALUES, PROBABILITIES)

    assert blended == pytest.approx(0.75 * MEAN + 0.25 * tail)
    assert MEAN < blended < tail


def test_full_weight_is_the_tail_alone():
    assert value_of_sample(
        CVaR(alpha=0.8, weight=1.0), VALUES, PROBABILITIES
    ) == pytest.approx(30.0)


def test_no_weight_is_the_mean_exactly():
    assert value_of_sample(
        CVaR(alpha=0.8, weight=0.0), VALUES, PROBABILITIES
    ) == pytest.approx(MEAN)

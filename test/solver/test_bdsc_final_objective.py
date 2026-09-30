"""The objective BDSC reports is the cost its first-stage solution incurs.

Column generation solves each subproblem with the coupling variables free,
so its last y answers an x of the subproblem's own choosing. Reporting
that y alongside the master's x paired a decision with a recourse it
cannot have, and the objective could come out below the optimum.

The instance is Caroe & Schultz's (as in examples/bdsc/cs.py) with r = 2:

    min 3x - E[2y]   s.t.  x in [l, 1],  y in {0, 1},  y/2 <= x - h_s

y_s = 1 is open only once x >= h_s + 1/2, so the recourse is a step
function and its value at any x can be written down directly.
"""

import logging

import pyomo.environ as pyo
import pytest

from pyodsp.model.sp import StochasticProgram

SOLVER = "appsi_highs"
R = 2
DELTA = 1 / 32 / (1 + R / 2)
LOWER = 1 / 4 - DELTA
H = [DELTA * s if s <= R / 2 else 1 / 4 - DELTA * (s - R / 2) for s in range(1, R + 1)]
OPTIMUM = 0.203125


def cost_at(x: float) -> float:
    """3x + E[Q(x)], evaluated by hand."""
    recourse = [-2.0 if x >= h + 0.5 - 1e-9 else 0.0 for h in H]
    return 3 * x + sum(recourse) / len(recourse)


def caroe_schultz(tmp_path, **kwargs):
    kwargs.setdefault("log_level", logging.CRITICAL)
    sp = StochasticProgram(
        "cs",
        sense="min",
        solver=SOLVER,
        recourse_bound=-2.01,
        output_dir=tmp_path,
        **kwargs,
    )

    @sp.first_stage
    def first_stage(m):
        m.x = pyo.Var(bounds=(LOWER, 1))
        return 3 * m.x

    @sp.recourse
    def recourse(m, state, scenario):
        m.y = pyo.Var(within=pyo.Binary)
        m.c1 = pyo.Constraint(expr=-m.y / 2 >= scenario["h"] - state.x)
        return -2 * m.y

    sp.set_scenarios({f"s{s}": {"h": h} for s, h in enumerate(H, 1)})
    return sp


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("max_iteration", [2, 4, 12])
def test_the_reported_objective_is_the_cost_of_the_reported_solution(
    tmp_path, max_iteration
):
    """Stopped early, the answer is not optimal, but it is still an answer:
    the objective must be what its x costs. These limits all stop where the
    subproblems' own x had let y = 1 that the master's x does not."""
    result = caroe_schultz(tmp_path, method="bdsc", max_iteration=max_iteration).solve()

    x = result.first_stage_flat["x"]
    assert result.objective == pytest.approx(cost_at(x), abs=1e-6)
    assert result.objective >= OPTIMUM - 1e-6


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_converged_run_reports_the_optimum(tmp_path):
    result = caroe_schultz(tmp_path, method="bdsc").solve()

    assert result.objective == pytest.approx(OPTIMUM, abs=1e-6)
    assert result.objective == pytest.approx(
        cost_at(result.first_stage_flat["x"]), abs=1e-6
    )

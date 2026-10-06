"""How the BDSC root decides it is done, and what it hands back.

Following van der Laan & Romeijnders (Algorithm 1 and Section 2.2), the
upper bound is the best solution evaluated so far, not the latest one, and
the run stops once a scaled cut no longer lifts the outer approximation at
the master's solution — returning that best solution either way.

Without these, Caroe & Schultz with r = 8 passed a solution within 0.2% of
the optimum at iteration 6, reached the optimal bound by iteration 9, and
then repeated one master solution until the iteration limit (977
iterations, ~10 minutes), reporting that solution's much worse cost.
"""

import logging

import pyomo.environ as pyo
import pytest

from pyodsp.alg.bm.cuts import CutList, OptimalityCut
from pyodsp.alg.const import STATUS_NOT_FINISHED, STATUS_OPTIMAL, STATUS_STALLED
from pyodsp.alg.params import BM_ABS_TOLERANCE
from pyodsp.dec.bdsc.alg_root_bm import BdScAlgRootBm
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig

BOUND = -2.0


def make_root():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, 1))
    model.obj = pyo.Objective(expr=3 * model.x, sense=pyo.minimize)
    alg = BdScAlgRootBm(PyomoSolver(model, SolverConfig("appsi_highs"), [model.x]))
    alg.set_logger(0, 0, logging.CRITICAL)
    alg.build([[1]], {1: 1.0}, {1: BOUND})
    return alg, model


def cut(rhs, slope):
    """theta >= rhs - slope * x"""
    return [
        CutList(
            [OptimalityCut(coeffs={0: slope}, rhs=rhs, objective_value=0.0, info={})]
        )
    ]


# -- the incumbent ----------------------------------------------------------


def test_the_incumbent_is_the_best_solution_seen_not_the_latest():
    alg, _ = make_root()

    alg._record_incumbent([0.2], 0.9)
    alg._record_incumbent([0.5], 0.4)
    alg._record_incumbent([0.8], 0.7)
    alg._record_incumbent([0.9], None)  # infeasible trial: no cost to compare

    assert alg.best_objective == 0.4
    assert alg.best_solution == [0.5]


def test_the_final_message_carries_the_incumbent_and_leaves_it_on_the_master():
    """The reported first stage and its cost are read off the master's
    variables, so the incumbent has to be put back there too."""
    alg, model = make_root()
    model.x.set_value(0.8)  # where the master last was
    alg._record_incumbent([0.5], 0.4)

    message = alg.get_final_dn_message()

    assert message.get_solution() == [0.5]
    assert model.x.value == 0.5


def test_without_an_incumbent_the_final_message_is_the_masters_solution():
    alg, model = make_root()
    model.x.set_value(0.8)

    assert alg.get_final_dn_message().get_solution() == [0.8]


# -- stopping ---------------------------------------------------------------


def test_the_gap_closes_against_the_best_incumbent():
    alg, _ = make_root()
    alg.bm.obj_bound = [0.4 - 1e-9]
    alg.bm.obj_val = [0.9]  # the latest trial is far worse
    alg._record_incumbent([0.5], 0.4)

    assert alg._termination_check(improvement=1.0) == STATUS_OPTIMAL


def test_an_open_gap_with_a_cut_that_still_improves_keeps_going():
    alg, _ = make_root()
    alg.bm.obj_bound = [0.2]
    alg._record_incumbent([0.5], 0.4)

    assert alg._termination_check(improvement=10 * BM_ABS_TOLERANCE) == (
        STATUS_NOT_FINISHED
    )


def test_a_cut_that_no_longer_improves_stops_the_run():
    alg, _ = make_root()
    alg.bm.obj_bound = [0.2]
    alg._record_incumbent([0.5], 0.4)

    assert alg._termination_check(improvement=0.1 * BM_ABS_TOLERANCE) == (
        STATUS_STALLED
    )


def test_the_improvement_is_the_cuts_lift_over_theta_at_the_trial_point():
    alg, _ = make_root()
    alg.bm.run_step(None)  # theta rests on its bound
    assert sum(alg.bm.get_theta_value()) == pytest.approx(BOUND)

    lifts = alg._improvement_at([0.5], cut(rhs=-1.0, slope=1.0))
    stays = alg._improvement_at([0.5], cut(rhs=-1.5, slope=1.0))

    assert lifts == pytest.approx(-1.5 - BOUND)  # -1.0 - 0.5 = -1.5 at x = 0.5
    assert stays == pytest.approx(-2.0 - BOUND)  # touches theta: no lift

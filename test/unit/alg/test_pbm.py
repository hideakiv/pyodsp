import time
from unittest.mock import MagicMock

import numpy as np

from pyodsp.alg.bm.pbm import ProximalBundleMethod
from pyodsp.alg.bm.cuts import CutList, OptimalityCut, FeasibilityCut
from pyodsp.alg.const import STATUS_NOT_FINISHED, STATUS_OPTIMAL


class FakeCpm:
    def __init__(self, relaxed_objective=0.0, current_solution=None, is_minimize=True):
        self.relaxed_objective = relaxed_objective
        self.current_solution = current_solution or []
        self._is_minimize = is_minimize

    def get_relaxed_objective(self):
        return self.relaxed_objective

    def get_current_solution(self):
        return self.current_solution

    def is_minimize(self):
        return self._is_minimize


def make_pbm(is_minimize=True, relaxed_objective=0.0, current_solution=None, center=None):
    pbm = ProximalBundleMethod.__new__(ProximalBundleMethod)
    pbm.cpm = FakeCpm(relaxed_objective, current_solution, is_minimize)
    pbm.max_iteration = 1000
    pbm.iteration = 0
    pbm.obj_bound = []
    pbm.obj_val = []
    pbm.status = STATUS_NOT_FINISHED
    pbm.start_time = time.time()
    pbm.penalty = 1.0
    pbm.center_val = []
    pbm.iter_since_update = 0
    pbm.e_v = np.inf
    pbm.logger = MagicMock()
    pbm.center = center or []
    pbm.num_cuts = 1
    pbm.subobj_bounds = None
    return pbm


def test_improved_returns_false_when_no_center():
    pbm = make_pbm()
    pbm.center_val = []
    assert pbm._improved() is False


def test_improved_minimize_true_when_sufficient_decrease():
    pbm = make_pbm(is_minimize=True, relaxed_objective=5.0)
    pbm.center_val = [10.0]
    pbm.obj_val = [6.0]

    assert pbm._improved() is True


def test_improved_minimize_false_when_insufficient_decrease():
    # predicted_diff = 5.0 - 10.0 = -5.0; threshold = 10.0 + 0.1 * -5.0 = 9.5
    pbm = make_pbm(is_minimize=True, relaxed_objective=5.0)
    pbm.center_val = [10.0]
    pbm.obj_val = [9.99]

    assert pbm._improved() is False


def test_improved_maximize_true_when_sufficient_increase():
    pbm = make_pbm(is_minimize=False, relaxed_objective=15.0)
    pbm.center_val = [10.0]
    pbm.obj_val = [14.0]

    assert pbm._improved() is True


def test_get_alpha_returns_none_when_obj_val_missing():
    pbm = make_pbm(current_solution=[0.0], center=[0.0])
    pbm.obj_val = [None]
    pbm.center_val = [1.0]

    assert pbm._get_alpha([CutList()]) is None


def test_get_alpha_returns_none_on_feasibility_cut():
    pbm = make_pbm(current_solution=[1.0], center=[0.0])
    pbm.obj_val = [5.0]
    pbm.center_val = [10.0]
    cuts_list = [CutList([FeasibilityCut(coeffs={0: 1.0}, rhs=0.0, info={})])]

    assert pbm._get_alpha(cuts_list) is None


def test_get_alpha_computes_value_for_optimality_cuts():
    pbm = make_pbm(current_solution=[2.0], center=[0.0])
    pbm.obj_val = [5.0]
    pbm.center_val = [10.0]
    cuts_list = [
        CutList(
            [OptimalityCut(coeffs={0: 3.0}, rhs=0.0, info={}, objective_value=0.0)]
        )
    ]

    # center_val - obj_val + inner(grad, d) = 10 - 5 + 3*(2-0) = 11
    assert pbm._get_alpha(cuts_list) == 11.0


def test_termination_check_optimal_when_predicted_diff_small():
    pbm = make_pbm(relaxed_objective=10.0)
    pbm.center_val = [10.0]
    pbm.obj_val = [10.0]
    pbm.obj_bound = [10.0]

    assert pbm._termination_check() is True
    assert pbm.status == STATUS_OPTIMAL


def test_termination_check_not_optimal_when_predicted_diff_large():
    pbm = make_pbm(relaxed_objective=100.0)
    pbm.center_val = [10.0]
    pbm.obj_val = [10.0]
    pbm.obj_bound = [10.0]

    assert pbm._termination_check() is False


def test_termination_check_false_when_no_center_val():
    pbm = make_pbm()
    pbm.center_val = []

    assert pbm._termination_check() is False


def test_serious_step_penalty_update_halves_penalty_after_many_stale_iterations():
    # predicted_diff = 0; obj_val > center_val so penalty_too_large is False,
    # taking the `iter_since_update > 3` branch.
    pbm = make_pbm(relaxed_objective=10.0, current_solution=[1.0])
    pbm.center_val = [10.0]
    pbm.obj_val = [11.0]
    pbm.penalty = 2.0
    pbm.iter_since_update = 4
    pbm.e_v = 0.0
    pbm._update_center = MagicMock()

    pbm._serious_step_penalty_update()

    assert pbm.penalty == 1.0
    assert pbm.iter_since_update == 1
    pbm._update_center.assert_called_once_with([1.0])


def test_serious_step_penalty_update_shrinks_penalty_when_too_large():
    # predicted_diff = 8.0 - 10.0 = -2.0; threshold = 10 + 0.5*-2 = 9.0
    # obj_val=8.5 <= 9.0 => penalty_too_large True, with iter_since_update > 0.
    pbm = make_pbm(relaxed_objective=8.0, current_solution=[1.0])
    pbm.center_val = [10.0]
    pbm.obj_val = [8.5]
    pbm.penalty = 1.0
    pbm.iter_since_update = 1
    pbm.e_v = 0.0
    pbm._update_center = MagicMock()

    pbm._serious_step_penalty_update()

    assert pbm.penalty == 0.5
    assert pbm.iter_since_update == 1
    pbm._update_center.assert_called_once()


def test_null_step_penalty_update_decrements_iter_since_update_by_default():
    pbm = make_pbm(relaxed_objective=10.0, current_solution=[1.0], center=[0.0])
    pbm.center_val = [10.0]
    pbm.obj_val = [10.0]
    pbm.penalty = 1.0
    pbm.iter_since_update = 0
    pbm.e_v = 100.0
    pbm._update_center = MagicMock()

    pbm._null_step_penalty_update([CutList()])

    assert pbm.penalty == 1.0
    assert pbm.iter_since_update == -1
    pbm._update_center.assert_not_called()

import time
from unittest.mock import MagicMock

from pyodsp.alg.bm.rbm import RestrictedBundleMethod
from pyodsp.alg.const import STATUS_NOT_FINISHED, STATUS_OPTIMAL


class FakeCpm:
    def __init__(self, theta_values=None, is_minimize=True):
        self.theta_values = theta_values or []

        self._is_minimize = is_minimize

    def get_theta_value(self, idx):
        return self.theta_values[idx]

    def is_minimize(self):
        return self._is_minimize

    def get_sign(self):
        return 1.0 if self._is_minimize else -1.0


def make_rbm(is_minimize=True, theta_values=None, num_cuts=1, subobj_bounds=None):
    rbm = RestrictedBundleMethod.__new__(RestrictedBundleMethod)
    rbm.cpm = FakeCpm(theta_values, is_minimize)
    rbm.max_iteration = 1000
    rbm.iteration = 0
    rbm.obj_bound = []
    rbm.obj_val = []
    rbm.status = STATUS_NOT_FINISHED
    rbm.start_time = time.time()
    rbm.penalty = 1.0
    rbm.center_val = []
    rbm.logger = MagicMock()
    rbm.num_cuts = num_cuts
    rbm.subobj_bounds = subobj_bounds
    return rbm


def test_improved_returns_false_when_no_center():
    rbm = make_rbm()
    rbm.center_val = []
    assert rbm._improved() is False


def test_improved_minimize_true_when_obj_val_lower():
    rbm = make_rbm(is_minimize=True)
    rbm.center_val = [10.0]
    rbm.obj_val = [9.0]
    assert rbm._improved() is True


def test_improved_minimize_false_when_obj_val_higher():
    rbm = make_rbm(is_minimize=True)
    rbm.center_val = [10.0]
    rbm.obj_val = [11.0]
    assert rbm._improved() is False


def test_improved_maximize_true_when_obj_val_higher():
    rbm = make_rbm(is_minimize=False)
    rbm.center_val = [10.0]
    rbm.obj_val = [11.0]
    assert rbm._improved() is True


def test_termination_check_false_when_no_center_val():
    rbm = make_rbm()
    rbm.center_val = []
    assert rbm._termination_check() is False


def test_termination_check_optimal_when_gap_small():
    rbm = make_rbm()
    rbm.center_val = [10.0]
    rbm.obj_bound = [10.0 + 1e-9]
    rbm.obj_val = [10.0]

    assert rbm._termination_check() is True
    assert rbm.status == STATUS_OPTIMAL
    # padding branch: obj_bound/obj_val/center_val all stay in sync
    assert len(rbm.center_val) == len(rbm.obj_bound)
    assert len(rbm.obj_val) == len(rbm.obj_bound)


def test_termination_check_not_optimal_when_gap_large():
    rbm = make_rbm()
    rbm.center_val = [10.0]
    rbm.obj_bound = [20.0]
    rbm.obj_val = [10.0]

    assert rbm._termination_check() is False
    assert rbm.status == STATUS_NOT_FINISHED


def test_termination_check_respects_bound_gap_short_circuit():
    rbm = make_rbm(theta_values=[1.0], num_cuts=1, subobj_bounds=[1.0])
    rbm.center_val = [10.0]
    rbm.obj_bound = [10.0]
    rbm.obj_val = [10.0]

    assert rbm._termination_check() is False

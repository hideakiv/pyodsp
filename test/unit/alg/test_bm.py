import time
from unittest.mock import MagicMock

from pyodsp.alg.bm.bm import BundleMethod
from pyodsp.alg.const import (
    STATUS_NOT_FINISHED,
    STATUS_OPTIMAL,
    STATUS_MAX_ITERATION,
    STATUS_TIME_LIMIT,
)
from pyodsp.alg.bm import bm as bm_module


class FakeCpm:
    def __init__(self, theta_values):
        self.theta_values = theta_values
        self.purge_calls = 0
        self.increment_calls = 0

    def get_theta_value(self, idx):
        return self.theta_values[idx]

    def increment_cuts(self):
        self.increment_calls += 1

    def purge_cuts(self):
        self.purge_calls += 1


def make_bm(theta_values=None, num_cuts=1, subobj_bounds=None, max_iteration=1000):
    bm = BundleMethod.__new__(BundleMethod)
    bm.cpm = FakeCpm(theta_values or [0.0] * num_cuts)
    bm.max_iteration = max_iteration
    bm.iteration = 0
    bm.obj_bound = []
    bm.obj_val = []
    bm.status = STATUS_NOT_FINISHED
    bm.start_time = time.time()
    bm.num_cuts = num_cuts
    bm.subobj_bounds = subobj_bounds if subobj_bounds is not None else [0.0] * num_cuts
    bm.logger = MagicMock()
    return bm


def test_termination_check_max_iteration():
    bm = make_bm(max_iteration=3)
    bm.iteration = 3

    assert bm._termination_check() is True
    assert bm.status == STATUS_MAX_ITERATION


def test_termination_check_time_limit(monkeypatch):
    monkeypatch.setattr(bm_module, "BM_TIME_LIMIT", 0)
    bm = make_bm()
    bm.start_time = time.time() - 1

    assert bm._termination_check() is True
    assert bm.status == STATUS_TIME_LIMIT


def test_termination_check_false_when_no_obj_val_yet():
    bm = make_bm()
    bm.obj_val = []
    bm.obj_bound = []

    assert bm._termination_check() is False
    assert bm.status == STATUS_NOT_FINISHED


def test_termination_check_false_when_obj_val_is_none():
    bm = make_bm()
    bm.obj_val = [None]
    bm.obj_bound = [1.0]

    assert bm._termination_check() is False


def test_termination_check_false_when_bound_gap_within_tolerance():
    bm = make_bm(theta_values=[1.0], subobj_bounds=[1.0])
    bm.obj_val = [10.0]
    bm.obj_bound = [10.0]

    assert bm._termination_check() is False
    assert bm.status == STATUS_NOT_FINISHED


def test_termination_check_optimal_when_relative_gap_small():
    bm = make_bm(theta_values=[5.0], subobj_bounds=[0.0])
    bm.obj_val = [10.0]
    bm.obj_bound = [10.0 + 1e-9]

    assert bm._termination_check() is True
    assert bm.status == STATUS_OPTIMAL


def test_termination_check_not_optimal_when_relative_gap_large():
    bm = make_bm(theta_values=[5.0], subobj_bounds=[0.0])
    bm.obj_val = [10.0]
    bm.obj_bound = [20.0]

    assert bm._termination_check() is False
    assert bm.status == STATUS_NOT_FINISHED


def test_termination_check_uses_absolute_tolerance_when_obj_val_near_zero():
    bm = make_bm(theta_values=[5.0], subobj_bounds=[0.0])
    bm.obj_val = [0.0]
    bm.obj_bound = [1.0]

    assert bm._termination_check() is False


def test_increment_purges_every_purge_freq_iterations(monkeypatch):
    monkeypatch.setattr(bm_module, "BM_PURGE_FREQ", 2)
    bm = make_bm()

    bm._increment()
    assert bm.cpm.purge_calls == 0
    bm._increment()
    assert bm.cpm.purge_calls == 1
    assert bm.cpm.increment_calls == 2

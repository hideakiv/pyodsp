from unittest.mock import MagicMock

import numpy as np
import pytest
import scipy.stats as st

from fakes import FakeAlgRoot, FakeAlgLeaf, FakeLogger

from pyodsp.dec.graph.lattice import Lattice
from pyodsp.dec.node.dec_node import DecNodeParent, DecNodeChild, DecNodeInner


def make_lattice(is_minimize=True, sample_size=5, confidence_level=0.95):
    lattice = Lattice.__new__(Lattice)
    lattice.logger = FakeLogger()
    lattice.is_minimize = is_minimize
    lattice.sample_size = sample_size
    lattice.confidence_level = confidence_level
    lattice.prev_samples = None
    lattice._start_time = 0.0
    # _termination records each convergence test, which needs somewhere to
    # put it and the root's sense to report it in.
    lattice.simulation_rounds = []
    lattice.root = DecNodeParent(idx="0-0", alg_root=FakeAlgRoot())
    lattice.gap_tolerance = 1e-2
    lattice.stable_tolerance = 1e-3
    lattice.stall_iterations = 3
    lattice.stall_tolerance = 1e-4
    lattice.bound_history = []
    lattice.stop_reason = None
    lattice.stop_message = None
    return lattice


def test_verify_nodes_rejects_multiple_stage_zero_nodes():
    root1 = DecNodeParent(idx=0, alg_root=FakeAlgRoot())
    root2 = DecNodeParent(idx=1, alg_root=FakeAlgRoot())
    leaf = DecNodeChild(idx=2, alg_leaf=FakeAlgLeaf())

    with pytest.raises(ValueError, match="Number of nodes is 2 in stage 0"):
        Lattice([[root1, root2], [leaf]], FakeLogger(), filedir=None, max_iteration=1)


def test_verify_nodes_rejects_leaf_at_stage_zero():
    leaf = DecNodeChild(idx=0, alg_leaf=FakeAlgLeaf())
    other_leaf = DecNodeChild(idx=1, alg_leaf=FakeAlgLeaf())

    with pytest.raises(ValueError, match="Stage 0 must be root node"):
        Lattice([[leaf], [other_leaf]], FakeLogger(), filedir=None, max_iteration=1)


def test_verify_nodes_rejects_root_at_last_stage():
    root = DecNodeParent(idx=0, alg_root=FakeAlgRoot())
    root_at_end = DecNodeParent(idx=1, alg_root=FakeAlgRoot())

    with pytest.raises(ValueError, match="Stage 1 must be leaf node"):
        Lattice([[root], [root_at_end]], FakeLogger(), filedir=None, max_iteration=1)


def test_verify_nodes_rejects_non_inner_at_middle_stage():
    root = DecNodeParent(idx=0, alg_root=FakeAlgRoot())
    middle_leaf = DecNodeChild(idx=1, alg_leaf=FakeAlgLeaf())
    leaf = DecNodeChild(idx=2, alg_leaf=FakeAlgLeaf())

    with pytest.raises(ValueError, match="Stage 1 must be inner node"):
        Lattice(
            [[root], [middle_leaf], [leaf]], FakeLogger(), filedir=None, max_iteration=1
        )


def test_verify_nodes_accepts_valid_three_stage_lattice(tmp_path):
    root = DecNodeParent(idx=0, alg_root=FakeAlgRoot())
    middle = DecNodeInner(idx=1, alg_root=FakeAlgRoot(), alg_leaf=FakeAlgLeaf())
    leaf = DecNodeChild(idx=2, alg_leaf=FakeAlgLeaf())

    lattice = Lattice(
        [[root], [middle], [leaf]], FakeLogger(), filedir=tmp_path, max_iteration=1
    )

    assert lattice.root is root
    assert lattice.leaves == [leaf]


def test_run_forwards_raises_when_multipliers_do_not_sum_to_one():
    lattice = make_lattice()
    lattice.num_stages = 2
    fake_root = MagicMock()
    fake_root.get_idx.return_value = 0
    fake_root.get_children.return_value = [1, 2]
    fake_root.get_multiplier.side_effect = lambda idx: {1: 0.5, 2: 0.6}[idx]
    lattice.root = fake_root
    lattice.nodes = {}

    with pytest.raises(ValueError, match="must sum to 1"):
        lattice._run_forwards(np.random.default_rng(0))


def _upper_limit(samples):
    return st.t.interval(
        confidence=0.95,
        df=len(samples) - 1,
        loc=np.mean(samples),
        scale=st.sem(samples),
    )[1]


# -- gap ----------------------------------------------------------------------


def test_the_gap_rule_stops_when_the_upper_limit_is_within_tolerance_of_the_bound():
    objectives = [9.0, 10.0, 11.0, 10.0, 10.0]
    lattice = make_lattice(sample_size=len(objectives))
    lattice._run_forwards = MagicMock(side_effect=objectives)
    upper = _upper_limit(objectives)

    assert lattice._termination(bound=upper / 1.005) is True
    assert lattice.stop_reason == "gap"


def test_the_gap_rule_waits_while_the_upper_limit_is_further_off():
    objectives = [9.0, 10.0, 11.0, 10.0, 10.0]
    lattice = make_lattice(sample_size=len(objectives))
    lattice._run_forwards = MagicMock(side_effect=objectives)
    upper = _upper_limit(objectives)

    assert lattice._termination(bound=upper / 1.05) is False
    assert lattice.stop_reason is None
    assert lattice.prev_samples == objectives


def test_the_gap_rule_is_one_sided():
    """An upper limit already below the (lower) bound has met it: the old
    two-sided test treated a large overshoot as not converged."""
    objectives = [9.0, 10.0, 11.0, 10.0, 10.0]
    lattice = make_lattice(sample_size=len(objectives))
    lattice._run_forwards = MagicMock(side_effect=objectives)

    assert lattice._termination(bound=1e3) is True
    assert lattice.stop_reason == "gap"


# -- stability ----------------------------------------------------------------


def test_the_stable_rule_stops_when_no_path_changed():
    objectives = [9.0, 10.0, 11.0, 10.0, 10.0]
    lattice = make_lattice(sample_size=len(objectives))
    lattice.prev_samples = list(objectives)
    lattice._run_forwards = MagicMock(side_effect=objectives)

    assert lattice._termination(bound=1.0) is True
    assert lattice.stop_reason == "stable"


def test_the_stable_rule_stops_when_the_change_is_small_both_ways():
    prev = [9.0, 10.0, 11.0, 10.0, 10.0]
    objectives = [9.001, 9.999, 11.0005, 10.0, 9.9995]
    lattice = make_lattice(sample_size=len(objectives))
    lattice.prev_samples = list(prev)
    lattice._run_forwards = MagicMock(side_effect=objectives)

    assert lattice._termination(bound=1.0) is True
    assert lattice.stop_reason == "stable"


def test_a_policy_that_got_worse_on_the_sample_does_not_stop_the_run():
    """The defect this rule replaces: it stopped when the upper end of the
    improvement's interval was below 1e-3, which a clear *deterioration*
    satisfies. Cuts still being added routinely make the fixed sample
    costlier for a while, and runs stopped with their bound still rising."""
    prev = [10.0, 10.0, 10.0, 10.0, 10.0]
    objectives = [12.0, 12.5, 11.8, 12.2, 12.1]
    lattice = make_lattice(sample_size=len(objectives))
    lattice.prev_samples = list(prev)
    lattice._run_forwards = MagicMock(side_effect=objectives)

    assert lattice._termination(bound=1.0) is False
    assert lattice.stop_reason is None
    assert lattice.prev_samples == objectives


def test_a_policy_that_still_improves_does_not_stop_the_run():
    prev = [12.0, 12.5, 11.8, 12.2, 12.1]
    objectives = [10.0, 10.0, 10.0, 10.0, 10.0]
    lattice = make_lattice(sample_size=len(objectives))
    lattice.prev_samples = list(prev)
    lattice._run_forwards = MagicMock(side_effect=objectives)

    assert lattice._termination(bound=1.0) is False


# -- stall --------------------------------------------------------------------


def test_the_stall_rule_needs_a_full_window_of_flat_bounds():
    lattice = make_lattice()  # window 3, tolerance 1e-4

    flags = [lattice._bound_stalled(b) for b in [5.0, 9.0, 10.0, 10.0, 10.0, 10.0]]

    assert flags == [False, False, False, False, False, True]


def test_the_stall_rule_ignores_a_bound_that_is_still_rising():
    lattice = make_lattice()

    flags = [lattice._bound_stalled(b) for b in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]

    assert not any(flags)


def test_the_stall_rule_can_be_turned_off():
    lattice = make_lattice()
    lattice.stall_iterations = 0

    assert not any(lattice._bound_stalled(10.0) for _ in range(10))


def test_a_stalled_run_still_ends_with_a_simulated_interval():
    objectives = [9.0, 10.0, 11.0, 10.0, 10.0]
    lattice = make_lattice(sample_size=len(objectives))
    lattice._run_forwards = MagicMock(side_effect=objectives)

    lattice._final_round(bound=9.5, iteration=41)

    assert lattice.stop_reason == "stall"
    assert lattice.simulation_rounds[-1].iteration == 41


def test_termination_records_the_interval_it_tested_with():
    """The interval is what stands against the bound, so it is kept.

    SDDP estimates that side by simulation rather than computing it, and
    a run that only logged the interval could not report or draw it.
    """
    objectives = [9.0, 10.0, 11.0, 10.0, 10.0]
    lattice = make_lattice(sample_size=len(objectives))
    lattice._run_forwards = MagicMock(side_effect=objectives)

    lattice._termination(bound=1e9, iteration=7)

    assert len(lattice.simulation_rounds) == 1
    round = lattice.simulation_rounds[0]
    assert round.iteration == 7
    assert round.sample_size == len(objectives)
    assert round.confidence_level == 0.95
    assert round.mean == pytest.approx(10.0)
    assert round.lower <= round.mean <= round.upper


def test_a_maximize_run_records_the_interval_in_its_own_units():
    """Negating swaps which confidence limit is the lower one."""
    objectives = [9.0, 10.0, 11.0, 10.0, 10.0]
    lattice = make_lattice(sample_size=len(objectives))
    lattice.root = DecNodeParent(idx="0-0", alg_root=FakeAlgRoot(sense_multiplier=-1.0))
    lattice._run_forwards = MagicMock(side_effect=objectives)

    lattice._termination(bound=1e9, iteration=3)

    round = lattice.simulation_rounds[0]
    assert round.mean == pytest.approx(-10.0)
    assert round.lower <= round.mean <= round.upper

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


def test_termination_converges_when_bound_matches_confidence_interval():
    objectives = [9.0, 10.0, 11.0, 10.0, 10.0]
    lattice = make_lattice(sample_size=len(objectives))
    lattice._run_forwards = MagicMock(side_effect=objectives)

    ci_d, ci_u = st.t.interval(
        confidence=0.95,
        df=len(objectives) - 1,
        loc=np.mean(objectives),
        scale=st.sem(objectives),
    )

    converged = lattice._termination(bound=ci_u)

    assert converged is True


def test_termination_does_not_converge_and_records_prev_samples():
    objectives = [9.0, 10.0, 11.0, 10.0, 10.0]
    lattice = make_lattice(sample_size=len(objectives))
    lattice._run_forwards = MagicMock(side_effect=objectives)

    converged = lattice._termination(bound=1e9)

    assert converged is False
    assert lattice.prev_samples == objectives


def test_termination_no_improvement_short_circuits_on_zero_diff():
    objectives = [9.0, 10.0, 11.0, 10.0, 10.0]
    lattice = make_lattice(sample_size=len(objectives))
    lattice.prev_samples = list(objectives)
    lattice._run_forwards = MagicMock(side_effect=objectives)

    # bound far from the CI keeps the primary convergence check False, so the
    # no-improvement branch (zero diff against identical prev_samples) decides.
    converged = lattice._termination(bound=1e9)

    assert converged is True


def test_termination_matches_reference_statistical_no_improvement_formula():
    prev_samples = [10.0, 10.0, 10.0, 10.0, 10.0]
    objectives = [9.0, 10.0, 11.0, 10.0, 10.0]
    lattice = make_lattice(sample_size=len(objectives))
    lattice.prev_samples = list(prev_samples)
    lattice._run_forwards = MagicMock(side_effect=objectives)

    sample_diffs = [prev_samples[i] - objectives[i] for i in range(len(objectives))]
    _, diff_ci_u = st.t.interval(
        confidence=0.95,
        df=len(sample_diffs) - 1,
        loc=np.mean(sample_diffs),
        scale=st.sem(sample_diffs),
    )
    from pyodsp.alg.params import SDDP_IMPROVE_TOLERANCE

    expected_no_improve = bool(diff_ci_u < SDDP_IMPROVE_TOLERANCE)

    converged = lattice._termination(bound=1e9)

    assert converged == expected_no_improve
    if not expected_no_improve:
        assert lattice.prev_samples == objectives

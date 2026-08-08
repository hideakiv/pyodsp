from fakes import FakeUpMessage

from pyodsp.alg.bm.cuts import OptimalityCut, FeasibilityCut
from pyodsp.dec.node.cut_aggregator import CutAggregator


def test_aggregates_single_group_of_optimality_cuts():
    groups = [[1, 2]]
    multipliers = {1: 0.5, 2: 0.5}
    aggregator = CutAggregator(groups, multipliers)

    cut1 = OptimalityCut(coeffs={0: 2.0}, rhs=4.0, info={"from": 1}, objective_value=10.0)
    cut2 = OptimalityCut(coeffs={0: 4.0}, rhs=8.0, info={"from": 2}, objective_value=20.0)
    up_messages = {1: FakeUpMessage(cut=cut1), 2: FakeUpMessage(cut=cut2)}

    aggregate_cuts = aggregator.get_aggregate_cuts(up_messages)

    assert len(aggregate_cuts) == 1
    result = aggregate_cuts[0][0]
    assert isinstance(result, OptimalityCut)
    assert result.coeffs == {0: 3.0}
    assert result.rhs == 6.0
    assert result.objective_value == 15.0


def test_keeps_info_when_group_has_single_optimality_cut():
    groups = [[1]]
    multipliers = {1: 1.0}
    aggregator = CutAggregator(groups, multipliers)

    cut = OptimalityCut(coeffs={0: 1.0}, rhs=1.0, info={"solution": [1.0]}, objective_value=1.0)
    up_messages = {1: FakeUpMessage(cut=cut)}

    aggregate_cuts = aggregator.get_aggregate_cuts(up_messages)

    assert aggregate_cuts[0][0].info == {"solution": [1.0]}


def test_drops_info_when_group_has_multiple_optimality_cuts():
    # info has no defined merge semantics across cuts from different children,
    # so an ambiguous aggregate must not silently expose one arbitrary child's
    # info as if it represented the whole group.
    groups = [[1, 2]]
    multipliers = {1: 1.0, 2: 1.0}
    aggregator = CutAggregator(groups, multipliers)

    cut1 = OptimalityCut(coeffs={0: 1.0}, rhs=1.0, info={"solution": [1.0]}, objective_value=1.0)
    cut2 = OptimalityCut(coeffs={0: 1.0}, rhs=1.0, info={"solution": [2.0]}, objective_value=1.0)
    up_messages = {1: FakeUpMessage(cut=cut1), 2: FakeUpMessage(cut=cut2)}

    aggregate_cuts = aggregator.get_aggregate_cuts(up_messages)

    assert aggregate_cuts[0][0].info == {}


def test_weights_coefficients_by_multiplier():
    groups = [[1, 2]]
    multipliers = {1: 1.0, 2: 3.0}
    aggregator = CutAggregator(groups, multipliers)

    cut1 = OptimalityCut(coeffs={0: 1.0}, rhs=1.0, info={}, objective_value=1.0)
    cut2 = OptimalityCut(coeffs={0: 1.0}, rhs=1.0, info={}, objective_value=1.0)
    up_messages = {1: FakeUpMessage(cut=cut1), 2: FakeUpMessage(cut=cut2)}

    aggregate_cuts = aggregator.get_aggregate_cuts(up_messages)

    result = aggregate_cuts[0][0]
    assert result.coeffs == {0: 4.0}
    assert result.rhs == 4.0
    assert result.objective_value == 4.0


def test_returns_all_feasibility_cuts_when_group_has_any():
    groups = [[1, 2]]
    multipliers = {1: 1.0, 2: 1.0}
    aggregator = CutAggregator(groups, multipliers)

    opt_cut = OptimalityCut(coeffs={0: 1.0}, rhs=1.0, info={}, objective_value=1.0)
    feas_cut = FeasibilityCut(coeffs={0: 1.0}, rhs=0.0, info={})
    up_messages = {1: FakeUpMessage(cut=opt_cut), 2: FakeUpMessage(cut=feas_cut)}

    aggregate_cuts = aggregator.get_aggregate_cuts(up_messages)

    result = aggregate_cuts[0]
    assert len(result) == 1
    assert isinstance(result[0], FeasibilityCut)
    assert result[0] is feas_cut


def test_returns_all_feasibility_cuts_when_multiple_present():
    groups = [[1, 2, 3]]
    multipliers = {1: 1.0, 2: 1.0, 3: 1.0}
    aggregator = CutAggregator(groups, multipliers)

    feas_cut1 = FeasibilityCut(coeffs={0: 1.0}, rhs=0.0, info={})
    feas_cut2 = FeasibilityCut(coeffs={0: 2.0}, rhs=1.0, info={})
    opt_cut = OptimalityCut(coeffs={0: 1.0}, rhs=1.0, info={}, objective_value=1.0)
    up_messages = {
        1: FakeUpMessage(cut=feas_cut1),
        2: FakeUpMessage(cut=opt_cut),
        3: FakeUpMessage(cut=feas_cut2),
    }

    aggregate_cuts = aggregator.get_aggregate_cuts(up_messages)

    result = aggregate_cuts[0]
    assert list(result) == [feas_cut1, feas_cut2]


def test_handles_multiple_independent_groups():
    groups = [[1], [2]]
    multipliers = {1: 1.0, 2: 1.0}
    aggregator = CutAggregator(groups, multipliers)

    cut1 = OptimalityCut(coeffs={0: 1.0}, rhs=1.0, info={}, objective_value=1.0)
    cut2 = OptimalityCut(coeffs={0: 2.0}, rhs=2.0, info={}, objective_value=2.0)
    up_messages = {1: FakeUpMessage(cut=cut1), 2: FakeUpMessage(cut=cut2)}

    aggregate_cuts = aggregator.get_aggregate_cuts(up_messages)

    assert len(aggregate_cuts) == 2
    assert aggregate_cuts[0][0].coeffs == {0: 1.0}
    assert aggregate_cuts[1][0].coeffs == {0: 2.0}

import itertools

import pytest

from pyodsp.alg.bm import cuts_manager as cuts_manager_module
from pyodsp.alg.bm.cuts_manager import CutsManager, CutInfo, NonPurgingCutsManager
from pyodsp.alg.bm.cuts import Cut, OptimalityCut, FeasibilityCut

_fake_constraint_ids = itertools.count()


class FakeConstraint:
    def __init__(self, lslack=1.0, uslack=1.0, name=None):
        self.deactivated = False
        self._lslack = lslack
        self._uslack = uslack
        self.name = name if name is not None else f"fake_constraint_{next(_fake_constraint_ids)}"

    def deactivate(self):
        self.deactivated = True

    def lslack(self):
        return self._lslack

    def uslack(self):
        return self._uslack


def make_cut_info(idx, coeffs, rhs, cut_cls=OptimalityCut, constraint=None, age=0):
    if cut_cls is OptimalityCut:
        cut = OptimalityCut(coeffs=coeffs, rhs=rhs, info={}, objective_value=0.0)
    else:
        cut = FeasibilityCut(coeffs=coeffs, rhs=rhs, info={})
    return CutInfo(
        constraint=constraint or FakeConstraint(),
        cut=cut,
        idx=idx,
        trial_point=[],
        age=age,
    )


def test_build_initializes_per_index_lists():
    manager = CutsManager()
    manager.build(3)
    assert manager.get_num_cuts() == 0
    assert manager.get_num_optimality(0) == 0
    assert manager.get_num_feasibility(0) == 0


def test_append_cut_counts_optimality_cuts():
    manager = CutsManager()
    manager.build(1)
    manager.append_cut(make_cut_info(0, {0: 1.0}, 1.0, OptimalityCut))
    assert manager.get_num_optimality(0) == 1
    assert manager.get_num_feasibility(0) == 0
    assert manager.get_num_cuts() == 1


def test_append_cut_counts_feasibility_cuts():
    manager = CutsManager()
    manager.build(1)
    manager.append_cut(make_cut_info(0, {0: 1.0}, 1.0, FeasibilityCut))
    assert manager.get_num_feasibility(0) == 1
    assert manager.get_num_optimality(0) == 0


def test_append_cut_raises_on_unrecognized_cut_type():
    manager = CutsManager()
    manager.build(1)
    plain_cut = Cut(coeffs={0: 1.0}, rhs=1.0, info={})
    cut_info = CutInfo(
        constraint=FakeConstraint(), cut=plain_cut, idx=0, trial_point=[], age=0
    )

    with pytest.raises(ValueError, match="Invalid cut type"):
        manager.append_cut(cut_info)


def test_append_cut_deduplicates_similar_cut():
    manager = CutsManager()
    manager.build(1)
    first = make_cut_info(0, {0: 1.0}, 1.0, OptimalityCut)
    manager.append_cut(first)

    duplicate_constraint = FakeConstraint()
    duplicate = make_cut_info(
        0, {0: 1.0}, 1.0, OptimalityCut, constraint=duplicate_constraint
    )
    manager.append_cut(duplicate)

    assert manager.get_num_cuts() == 1
    assert duplicate_constraint.deactivated is True


def test_append_cut_keeps_dissimilar_cut():
    manager = CutsManager()
    manager.build(1)
    manager.append_cut(make_cut_info(0, {0: 1.0}, 1.0, OptimalityCut))
    manager.append_cut(make_cut_info(0, {0: 2.0}, 5.0, OptimalityCut))

    assert manager.get_num_cuts() == 2


def test_increment_ages_slack_cut_when_more_than_one_active():
    manager = CutsManager()
    manager.build(1)
    slack_constraint = FakeConstraint(lslack=1.0, uslack=1.0)
    manager.append_cut(
        make_cut_info(0, {0: 1.0}, 1.0, OptimalityCut, constraint=slack_constraint)
    )
    manager.append_cut(make_cut_info(0, {0: 2.0}, 5.0, OptimalityCut))

    manager.increment()

    cuts = manager.get_cuts()[0]
    aged = [c for c in cuts if c.constraint is slack_constraint][0]
    assert aged.age == 1


def test_increment_resets_age_when_binding():
    manager = CutsManager()
    manager.build(1)
    binding_constraint = FakeConstraint(lslack=0.0, uslack=0.0)
    manager.append_cut(
        make_cut_info(
            0, {0: 1.0}, 1.0, OptimalityCut, constraint=binding_constraint, age=3
        )
    )
    manager.append_cut(make_cut_info(0, {0: 2.0}, 5.0, OptimalityCut))

    manager.increment()

    cuts = manager.get_cuts()[0]
    binding = [c for c in cuts if c.constraint is binding_constraint][0]
    assert binding.age == 0


def test_increment_does_not_age_single_active_cut():
    manager = CutsManager()
    manager.build(1)
    lone_constraint = FakeConstraint(lslack=1.0, uslack=1.0)
    manager.append_cut(
        make_cut_info(0, {0: 1.0}, 1.0, OptimalityCut, constraint=lone_constraint)
    )

    manager.increment()

    assert manager.get_cuts()[0][0].age == 0


class FakeModel:
    def __init__(self):
        self.deleted_names = []

    def del_component(self, name):
        self.deleted_names.append(name)


def test_purge_removes_cuts_past_max_age(monkeypatch):
    monkeypatch.setattr(cuts_manager_module, "BM_MAX_CUT_AGE", 2)
    manager = CutsManager()
    manager.build(1)
    old_constraint = FakeConstraint()
    manager.append_cut(
        make_cut_info(0, {0: 1.0}, 1.0, OptimalityCut, constraint=old_constraint, age=5)
    )
    manager.append_cut(make_cut_info(0, {0: 2.0}, 5.0, OptimalityCut, age=0))

    model = FakeModel()
    manager.purge(model)

    assert manager.get_num_cuts() == 1
    assert old_constraint.deactivated is True
    assert model.deleted_names == [old_constraint.name]


def test_purge_keeps_cuts_within_max_age(monkeypatch):
    monkeypatch.setattr(cuts_manager_module, "BM_MAX_CUT_AGE", 10)
    manager = CutsManager()
    manager.build(1)
    manager.append_cut(make_cut_info(0, {0: 1.0}, 1.0, OptimalityCut, age=1))

    model = FakeModel()
    manager.purge(model)

    assert manager.get_num_cuts() == 1
    assert model.deleted_names == []


def test_eliminate_cuts_removes_only_named_cuts():
    manager = CutsManager()
    manager.build(1)
    keep = FakeConstraint()
    remove = FakeConstraint()
    manager.append_cut(make_cut_info(0, {0: 1.0}, 1.0, OptimalityCut, constraint=keep))
    manager.append_cut(make_cut_info(0, {0: 2.0}, 5.0, OptimalityCut, constraint=remove))

    model = FakeModel()
    manager.eliminate_cuts(model, [remove.name])

    assert manager.get_num_cuts() == 1
    assert manager.get_cuts()[0][0].constraint is keep
    assert remove.deactivated is True
    assert keep.deactivated is False
    assert model.deleted_names == [remove.name]


def test_eliminate_cuts_ignores_unknown_names():
    manager = CutsManager()
    manager.build(1)
    manager.append_cut(make_cut_info(0, {0: 1.0}, 1.0, OptimalityCut))

    manager.eliminate_cuts(FakeModel(), ["does_not_exist"])

    assert manager.get_num_cuts() == 1


def test_eliminate_cuts_does_nothing_for_empty_names():
    manager = CutsManager()
    manager.build(1)
    manager.append_cut(make_cut_info(0, {0: 1.0}, 1.0, OptimalityCut))

    model = FakeModel()
    manager.eliminate_cuts(model, [])

    assert manager.get_num_cuts() == 1
    assert model.deleted_names == []


def test_non_purging_manager_increment_does_not_age_cuts():
    manager = NonPurgingCutsManager()
    manager.build(1)
    constraint = FakeConstraint(lslack=1.0, uslack=1.0)
    manager.append_cut(
        make_cut_info(0, {0: 1.0}, 1.0, OptimalityCut, constraint=constraint)
    )
    manager.append_cut(make_cut_info(0, {0: 2.0}, 5.0, OptimalityCut))

    manager.increment()

    assert manager.get_cuts()[0][0].age == 0


def test_non_purging_manager_purge_removes_nothing(monkeypatch):
    monkeypatch.setattr(cuts_manager_module, "BM_MAX_CUT_AGE", 1)
    manager = NonPurgingCutsManager()
    manager.build(1)
    manager.append_cut(make_cut_info(0, {0: 1.0}, 1.0, OptimalityCut, age=100))

    model = FakeModel()
    manager.purge(model)

    assert model.deleted_names == []
    assert manager.get_num_cuts() == 1


def test_non_purging_manager_still_eliminates_when_instructed():
    manager = NonPurgingCutsManager()
    manager.build(1)
    remove = FakeConstraint()
    manager.append_cut(make_cut_info(0, {0: 1.0}, 1.0, OptimalityCut, constraint=remove))

    manager.eliminate_cuts(FakeModel(), [remove.name])

    assert manager.get_num_cuts() == 0
    assert remove.deactivated is True


def test_purging_and_non_purging_managers_stay_synchronized(monkeypatch):
    # simulates the intended workflow (see BundleMethod.replace_cuts, which
    # LatticeMpi drives): a master CutsManager purges based on its own age
    # tracking, and a replica NonPurgingCutsManager resyncs by clearing
    # everything and mirroring the master's surviving cuts wholesale —
    # never deciding independently what's stale.
    monkeypatch.setattr(cuts_manager_module, "BM_MAX_CUT_AGE", 1)
    master = CutsManager()
    replica = NonPurgingCutsManager()
    master.build(1)
    replica.build(1)

    stale = FakeConstraint(lslack=1.0, uslack=1.0)
    fresh = FakeConstraint(lslack=0.0, uslack=0.0)
    for manager in (master, replica):
        manager.append_cut(
            make_cut_info(0, {0: 1.0}, 1.0, OptimalityCut, constraint=stale, age=5)
        )
        manager.append_cut(
            make_cut_info(0, {0: 2.0}, 5.0, OptimalityCut, constraint=fresh)
        )

    master.purge(FakeModel())

    survivors = [c for cuts in master.get_cuts() for c in cuts]
    replica.eliminate_cuts(
        FakeModel(), [c.constraint.name for cuts in replica.get_cuts() for c in cuts]
    )
    for cut_info in survivors:
        replica.append_cut(cut_info)

    assert [c.constraint for c in master.get_cuts()[0]] == [
        c.constraint for c in replica.get_cuts()[0]
    ]
    assert [c.constraint for c in master.get_cuts()[0]] == [fresh]

from pyodsp.alg.bm import cuts_manager as cuts_manager_module
from pyodsp.alg.bm.cuts_manager import CutsManager, CutInfo
from pyodsp.alg.bm.cuts import OptimalityCut, FeasibilityCut


class FakeConstraint:
    def __init__(self, lslack=1.0, uslack=1.0):
        self.deactivated = False
        self._lslack = lslack
        self._uslack = uslack
        self.name = "fake_constraint"

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
    def del_component(self, name):
        pass


def test_purge_removes_cuts_past_max_age(monkeypatch):
    monkeypatch.setattr(cuts_manager_module, "BM_MAX_CUT_AGE", 2)
    manager = CutsManager()
    manager.build(1)
    old_constraint = FakeConstraint()
    manager.append_cut(
        make_cut_info(0, {0: 1.0}, 1.0, OptimalityCut, constraint=old_constraint, age=5)
    )
    manager.append_cut(make_cut_info(0, {0: 2.0}, 5.0, OptimalityCut, age=0))

    manager.purge(FakeModel())

    assert manager.get_num_cuts() == 1
    assert old_constraint.deactivated is True


def test_purge_keeps_cuts_within_max_age(monkeypatch):
    monkeypatch.setattr(cuts_manager_module, "BM_MAX_CUT_AGE", 10)
    manager = CutsManager()
    manager.build(1)
    manager.append_cut(make_cut_info(0, {0: 1.0}, 1.0, OptimalityCut, age=1))

    manager.purge(FakeModel())

    assert manager.get_num_cuts() == 1

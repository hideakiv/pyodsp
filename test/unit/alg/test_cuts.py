from pyodsp.alg.bm.cuts import Cut, OptimalityCut, FeasibilityCut, CutList


def test_cut_holds_fields():
    cut = Cut(coeffs={0: 1.0, 1: -2.0}, rhs=3.0, info={"k": "v"})
    assert cut.coeffs == {0: 1.0, 1: -2.0}
    assert cut.rhs == 3.0
    assert cut.info == {"k": "v"}


def test_optimality_cut_adds_objective_value():
    cut = OptimalityCut(coeffs={0: 1.0}, rhs=1.0, info={}, objective_value=5.0)
    assert isinstance(cut, Cut)
    assert cut.objective_value == 5.0


def test_feasibility_cut_is_plain_cut():
    cut = FeasibilityCut(coeffs={0: 1.0}, rhs=1.0, info={})
    assert isinstance(cut, Cut)
    assert not isinstance(cut, OptimalityCut)


def test_optimality_and_feasibility_cut_are_distinguishable():
    opt = OptimalityCut(coeffs={}, rhs=0.0, info={}, objective_value=0.0)
    feas = FeasibilityCut(coeffs={}, rhs=0.0, info={})
    assert isinstance(opt, OptimalityCut)
    assert not isinstance(opt, FeasibilityCut)
    assert isinstance(feas, FeasibilityCut)
    assert not isinstance(feas, OptimalityCut)


def test_cutlist_defaults_to_empty():
    cl = CutList()
    assert list(cl) == []


def test_cutlist_wraps_given_items():
    cut = FeasibilityCut(coeffs={}, rhs=0.0, info={})
    cl = CutList([cut])
    assert list(cl) == [cut]
    assert isinstance(cl, list)


def test_cutlist_supports_list_operations():
    cl = CutList()
    cut = FeasibilityCut(coeffs={}, rhs=0.0, info={})
    cl.append(cut)
    assert len(cl) == 1
    assert cl[0] is cut

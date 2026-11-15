"""An inner stage's sddpsc leaf prices the next stage's cost-to-go.

A two-stage bdsc leaf prices only its own stage's cost. At an inner stage
of a lattice that leaves everything downstream out of the cut sent up, so
SddpScAlgLeafPyomo mirrors the node's root-role cuts onto its own copy of
the outgoing state and adds that theta to the recourse it prices -- while
still reporting only the stage's own cost as what a sample path pays.

The instance: incoming state x in [0, 1], outgoing state z in [x, 2],
stage cost 2z, and one mirrored cut theta >= 3 - z (theta floored at 0).
So Q(x) = min_z 2z + max(0, 3 - z) = 3 + x, with z = x, stage cost 2x.
"""

import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.dec.bdsc.alg_leaf_pyomo import BdScAlgLeafPyomo
from pyodsp.dec.bdsc.message import BdScDnMessage, BdScFinalDnMessage
from pyodsp.dec.sddpsc.alg_leaf_pyomo import SddpScAlgLeafPyomo
from pyodsp.alg.bm.cuts import CutList, OptimalityCut

SOLVER = "appsi_highs"
PARENT_FLOOR = -10.0
FUTURE_FLOOR = 0.0
X_HAT = 0.5


def make_model():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, 1))
    model.z = pyo.Var(bounds=(0, 2))
    model.link = pyo.Constraint(expr=model.z >= model.x)
    model.obj = pyo.Objective(expr=2 * model.z, sense=pyo.minimize)
    return model


def make_leaf(cls=SddpScAlgLeafPyomo, with_future=True):
    model = make_model()
    solver = PyomoSolver(model, SolverConfig(SOLVER), [model.x])
    kwargs = {"future_vars": [model.z]} if cls is SddpScAlgLeafPyomo else {}
    leaf = cls(solver, SolverConfig("ipopt"), 50, **kwargs)
    leaf.set_logger(idx=0, depth=1, level=0)
    leaf.build()
    if with_future and cls is SddpScAlgLeafPyomo:
        leaf.build_future([FUTURE_FLOOR])
    return leaf


def future_cut(rhs=3.0, coeff=1.0):
    """theta + coeff * z >= rhs, as the root role's bm would hold it."""
    return OptimalityCut(coeffs={0: coeff}, rhs=rhs, info={}, objective_value=rhs)


def dn_message(x=X_HAT):
    return BdScDnMessage(
        solution=[x],
        rho=PARENT_FLOOR,
        cut_list=[CutList([])],
        subobj_bounds=[PARENT_FLOOR],
        objective=0.0,
    )


def up(leaf, x=X_HAT):
    leaf.pass_dn_message(dn_message(x))
    return leaf.get_up_message()


def test_without_future_it_matches_bdsc_leaf():
    plain = up(make_leaf(BdScAlgLeafPyomo))
    unbuilt = up(make_leaf(with_future=False))

    assert unbuilt.get_objective() == pytest.approx(plain.get_objective())
    assert unbuilt.get_cut().rhs == pytest.approx(plain.get_cut().rhs)
    assert unbuilt.get_cut().coeffs == pytest.approx(plain.get_cut().coeffs)


def test_recourse_includes_the_future_but_the_sample_cost_does_not():
    leaf = make_leaf()
    leaf.set_future_cuts([CutList([future_cut()])])

    message = up(leaf)

    # Q(x_hat) = 3 + x_hat prices the future ...
    assert message.get_cut().objective_value == pytest.approx(3.0 + X_HAT)
    # ... but a sample path only pays this stage's 2 * z
    assert message.get_objective() == pytest.approx(2 * X_HAT)


def test_scaled_cut_is_valid_and_sees_the_future():
    leaf = make_leaf()
    leaf.set_future_cuts([CutList([future_cut()])])

    message = up(leaf)
    cut, tau = message.get_cut(), message.get_tau()

    def cut_value(x):
        # what the root role installs: theta >= (alpha - beta x) / (1 + tau)
        beta = cut.coeffs.get(0, 0.0)
        return (cut.rhs - beta * x) / (1 + tau)

    for x in (0.0, 0.25, X_HAT, 0.75, 1.0):
        assert cut_value(x) <= 3.0 + x + 1e-6
    # stage cost alone never exceeds 2 at x_hat's neighbourhood; a cut
    # above it can only have come from the future term
    assert cut_value(X_HAT) > 2 * X_HAT + 1e-6


def test_final_evaluation_reports_the_stage_cost():
    leaf = make_leaf()
    leaf.set_future_cuts([CutList([future_cut()])])
    up(leaf)

    leaf.pass_final_dn_message(BdScFinalDnMessage([X_HAT]))

    assert leaf.get_final_up_message().get_objective() == pytest.approx(2 * X_HAT)


def test_columns_survive_an_unchanged_future_and_retire_on_a_change():
    leaf = make_leaf()
    leaf.set_future_cuts([CutList([future_cut()])])
    up(leaf)
    columns = list(leaf._cgmp_cuts)
    assert columns

    leaf.set_future_cuts([CutList([future_cut()])])
    assert leaf._cgmp_cuts == columns

    leaf.set_future_cuts([CutList([future_cut(), future_cut(rhs=4.0, coeff=2.0)])])
    assert leaf._cgmp_cuts == []

    # dropping a cut lowers q at recorded columns too
    up(leaf)
    leaf.set_future_cuts([CutList([future_cut()])])
    assert leaf._cgmp_cuts == []


def test_future_cuts_are_replaced_not_accumulated():
    leaf = make_leaf()
    leaf.set_future_cuts([CutList([future_cut(rhs=10.0)])])
    leaf.set_future_cuts([CutList([future_cut()])])

    # with the rhs=10 cut gone, Q(x_hat) is back to 3 + x_hat
    assert up(leaf).get_cut().objective_value == pytest.approx(3.0 + X_HAT)


def test_misuse_is_refused():
    with pytest.raises(ValueError, match="future_vars"):
        model = make_model()
        leaf = SddpScAlgLeafPyomo(
            PyomoSolver(model, SolverConfig(SOLVER), [model.x]),
            SolverConfig("ipopt"),
        )
        leaf.build_future([0.0])

    with pytest.raises(ValueError, match="before build_future"):
        make_leaf(with_future=False).set_future_cuts([CutList([])])

    with pytest.raises(ValueError, match="cut group"):
        make_leaf().set_future_cuts([CutList([]), CutList([])])

    with pytest.raises(ValueError, match="bound"):
        make_leaf(with_future=False).build_future([None])


def test_a_change_after_the_trial_point_rebuilds_the_cgmp_without_columns():
    # Lattice's order: pass_dn_message (cgmp seeded with saved columns),
    # then the root role's cuts change on the backward pass, then
    # get_up_message. The seeded columns must not survive into it.
    leaf = make_leaf()
    leaf.set_future_cuts([CutList([future_cut()])])
    up(leaf)
    leaf.pass_dn_message(dn_message())
    assert leaf.cgmp.cpm.get_num_cuts() > 0

    leaf.set_future_cuts([CutList([future_cut(), future_cut(rhs=4.0, coeff=2.0)])])

    assert leaf.cgmp.cpm.get_num_cuts() == 0
    # theta >= 4 - 2z joins theta >= 3 - z: Q(x) = max(3 + x, 4) at z = x
    assert leaf.get_up_message().get_cut().objective_value == pytest.approx(4.0)

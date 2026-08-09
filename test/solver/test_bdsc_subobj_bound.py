"""The floor under the cgsp's theta is the root's, read once.

The root only works its subproblem bounds out in build(), which happens
after its children have reported theirs upward — so BdScInitDnMessage,
which goes down before that, cannot carry them. They ride on the dn
messages instead, and the leaf takes the first and holds it.
"""

import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.dec.bdsc.alg_leaf_pyomo import BdScAlgLeafPyomo
from pyodsp.dec.bdsc.message import BdScDnMessage

SOLVER = "appsi_highs"
PLACEHOLDER_FLOOR = -1e9


def make_leaf(depth=1):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(-100, 100))
    model.s = pyo.Var(domain=pyo.Reals, bounds=(0, 100))
    model.obj = pyo.Objective(expr=model.s, sense=pyo.minimize)
    model.con = pyo.Constraint(expr=model.s >= model.x)
    leaf = BdScAlgLeafPyomo(
        PyomoSolver(model, SolverConfig(SOLVER), [model.x]), SolverConfig("ipopt"), 20
    )
    leaf.set_logger(idx=0, depth=depth, level=0)
    leaf.build()
    return leaf


def message(subobj_bounds, solution=None):
    return BdScDnMessage(
        solution=solution if solution is not None else [5.0],
        rho=1.0,
        cut_list=None,
        subobj_bounds=subobj_bounds,
        objective=0.0,
    )


def theta_lb(leaf):
    return leaf.cgsp.cpm.solver.model._theta[0].lb


def test_build_leaves_a_placeholder_floor_until_a_trial_point_arrives():
    leaf = make_leaf()

    assert theta_lb(leaf) == PLACEHOLDER_FLOOR
    assert leaf._subobj_bound is None


def test_the_first_trial_point_installs_the_root_bound():
    leaf = make_leaf()

    leaf.pass_dn_message(message([-7.5]))

    assert theta_lb(leaf) == pytest.approx(-7.5)
    assert leaf._subobj_bound == pytest.approx(-7.5)


def test_later_trial_points_do_not_reinstall_it():
    leaf = make_leaf()
    leaf.pass_dn_message(message([-7.5]))

    # a different trial point carrying the same (constant) bound
    leaf.pass_dn_message(message([-7.5], solution=[9.0]))

    assert theta_lb(leaf) == pytest.approx(-7.5)


def test_a_changed_bound_is_refused():
    leaf = make_leaf()
    leaf.pass_dn_message(message([-7.5]))

    with pytest.raises(ValueError, match="subproblem bound changed"):
        leaf.pass_dn_message(message([-3.0]))


def test_an_unbounded_child_is_reported_clearly():
    # a leaf whose set_bound was never called leaves the root with None here,
    # which used to surface as a TypeError from summing None
    leaf = make_leaf()

    with pytest.raises(ValueError, match="needs a bound on every child"):
        leaf.pass_dn_message(message([None]))


def test_more_than_one_group_is_reported_clearly():
    leaf = make_leaf()

    with pytest.raises(ValueError, match="single group"):
        leaf.pass_dn_message(message([-1.0, -2.0]))

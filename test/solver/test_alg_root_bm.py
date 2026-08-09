import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.dec.bd.alg_root_bm import BdAlgRootBm
from pyodsp.dec.dd.alg_root_bm import DdAlgRootBm


def make_root(sense=pyo.minimize):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(-100, 100))
    model.obj = pyo.Objective(expr=model.x, sense=sense)
    solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])
    return BdAlgRootBm(solver), model


def test_build_computes_group_bound_when_all_members_bounded():
    root, model = make_root()
    root.set_logger(node_id=0, depth=0)

    root.build(groups=[[1, 2]], children_multipliers={1: 1.0, 2: 2.0}, children_bounds={1: 3.0, 2: 4.0})

    assert root.bm.subobj_bounds == [1.0 * 3.0 + 2.0 * 4.0]


def test_build_leaves_group_bound_none_when_any_member_unbounded():
    root, model = make_root()
    root.set_logger(node_id=0, depth=0)

    root.build(groups=[[1, 2]], children_multipliers={1: 1.0, 2: 1.0}, children_bounds={1: 3.0})

    assert root.bm.subobj_bounds == [None]
    # theta[0] should be unbounded below for minimize when no bound is known
    assert root.bm.cpm.get_solver().model._theta[0].lb is None


def test_build_computes_independent_bounds_per_group():
    root, model = make_root()
    root.set_logger(node_id=0, depth=0)

    root.build(
        groups=[[1], [2]],
        children_multipliers={1: 1.0, 2: 1.0},
        children_bounds={1: 5.0, 2: 7.0},
    )

    assert root.bm.subobj_bounds == [5.0, 7.0]


def test_purgeable_true_by_default_and_does_not_force_add():
    solver = PyomoSolver(*_root_model_and_vars())
    root = BdAlgRootBm(solver)

    assert root.bm.cpm.cuts_manager.__class__.__name__ == "CutsManager"
    assert root.bm.cpm.force is False


def test_purgeable_false_uses_non_purging_manager_and_forces_add():
    solver = PyomoSolver(*_root_model_and_vars())
    root = BdAlgRootBm(solver, purgeable=False)

    assert root.bm.cpm.cuts_manager.__class__.__name__ == "NonPurgingCutsManager"
    assert root.bm.cpm.force is True


def _root_model_and_vars():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(-100, 100))
    model.obj = pyo.Objective(expr=model.x, sense=pyo.minimize)
    return model, SolverConfig("appsi_highs"), [model.x]


def make_coupling_model_and_vars():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals)
    model.y = pyo.Var(domain=pyo.Reals)
    model.con = pyo.Constraint(expr=model.x + model.y <= 10)
    return model, {0: [model.x], 1: [model.y]}


def test_dd_alg_root_bm_rejects_objective_in_coupling_model():
    model, vars_dn = make_coupling_model_and_vars()
    model.obj = pyo.Objective(expr=model.x)

    with pytest.raises(ValueError, match="Objective should not be defined"):
        DdAlgRootBm(model, SolverConfig("appsi_highs"), vars_dn)


def test_dd_alg_root_bm_rejects_var_not_in_coupling_model():
    model, vars_dn = make_coupling_model_and_vars()
    stray_model = pyo.ConcreteModel()
    stray_model.z = pyo.Var()
    vars_dn[2] = [stray_model.z]

    with pytest.raises(ValueError, match="does not exist in varname_list"):
        DdAlgRootBm(model, SolverConfig("appsi_highs"), vars_dn)


def test_dd_alg_root_bm_rejects_uncoupled_leftover_vars():
    model, _ = make_coupling_model_and_vars()
    # only x is included in vars_dn; y is left uncoupled
    vars_dn = {0: [model.x]}

    with pytest.raises(ValueError, match="not coupled"):
        DdAlgRootBm(model, SolverConfig("appsi_highs"), vars_dn)


def test_dd_alg_root_bm_accepts_valid_coupling_and_defaults_to_bundle_method():
    model, vars_dn = make_coupling_model_and_vars()

    root = DdAlgRootBm(model, SolverConfig("appsi_highs"), vars_dn)

    from pyodsp.alg.bm.bm import BundleMethod

    assert type(root.bm) is BundleMethod


def test_dd_alg_root_bm_proximal_mode_uses_proximal_bundle_method():
    model, vars_dn = make_coupling_model_and_vars()

    root = DdAlgRootBm(model, SolverConfig("ipopt"), vars_dn, mode="proximal")

    from pyodsp.alg.bm.pbm import ProximalBundleMethod

    assert type(root.bm) is ProximalBundleMethod


def test_dd_alg_root_bm_rejects_invalid_mode():
    model, vars_dn = make_coupling_model_and_vars()

    with pytest.raises(ValueError, match="Invalid mode"):
        DdAlgRootBm(model, SolverConfig("appsi_highs"), vars_dn, mode="bogus")

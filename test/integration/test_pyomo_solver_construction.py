import pickle

import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig


def make_model(sense=pyo.minimize):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(0, 10))
    model.y = pyo.Var(domain=pyo.Reals, bounds=(0, 10))
    model.obj = pyo.Objective(expr=model.x + model.y, sense=sense)
    model.con = pyo.Constraint(expr=model.x + model.y <= 5)
    return model


def make_solver(sense=pyo.minimize, vars=None):
    model = make_model(sense)
    return PyomoSolver(
        model, SolverConfig("appsi_highs"), vars if vars is not None else [model.x, model.y]
    )


def test_get_objective_raises_when_no_objective_defined():
    model = pyo.ConcreteModel()
    model.x = pyo.Var()

    with pytest.raises(ValueError, match="Objective not found"):
        PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])


def test_is_minimize_true_for_minimize_sense():
    solver = make_solver(sense=pyo.minimize)
    assert solver.is_minimize() is True


def test_is_minimize_false_for_maximize_sense():
    solver = make_solver(sense=pyo.maximize)
    assert solver.is_minimize() is False


def test_get_original_objective_value_returns_zero_before_solve():
    solver = make_solver()
    # Not actually a bug: CuttingPlaneMethod.add_cuts() relies on this to
    # bootstrap obj_val to a real number on the very first call (before the
    # solver has ever solved), so ProximalBundleMethod/RestrictedBundleMethod
    # can seed center_val on their first run_step. Returning None here instead
    # reproduces the crash the source's FIXME comment refers to — see
    # pbm.py::_null_step_penalty_update, which indexes an empty center_val.
    assert solver.get_original_objective_value() == 0


def test_get_vars_returns_constructor_vars():
    model = make_model()
    solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])
    assert solver.get_vars() == [model.x]


def test_parent_objective_value_defaults_to_zero_and_is_settable():
    solver = make_solver()
    assert solver.get_parent_objective_value() == 0.0
    solver.set_parent_objective_value(3.5)
    assert solver.get_parent_objective_value() == 3.5


@pytest.mark.parametrize("fmt", ["lp", "mps", "nl"])
def test_save_model_writes_solver_format(tmp_path, fmt):
    solver = make_solver()
    solver.save_model(tmp_path, format=fmt)
    assert (tmp_path / f"model.{fmt}").exists()


def test_save_model_pickle_roundtrips(tmp_path):
    solver = make_solver()
    solver.save_model(tmp_path, format="pickle")
    with open(tmp_path / "model.pkl", "rb") as f:
        loaded = pickle.load(f)
    assert pyo.value(loaded.x.lb) == 0


def test_save_model_rejects_unknown_format(tmp_path):
    solver = make_solver()
    with pytest.raises(ValueError, match="Unsupported format"):
        solver.save_model(tmp_path, format="xml")


def test_load_model_pickle_roundtrips(tmp_path):
    solver = make_solver()
    solver.save_model(tmp_path, format="pickle")

    loaded = solver.load_model(tmp_path, format="pickle")

    assert isinstance(loaded, pyo.ConcreteModel)


@pytest.mark.parametrize("fmt", ["lp", "mps", "nl"])
def test_load_model_solver_formats_not_implemented(tmp_path, fmt):
    solver = make_solver()
    with pytest.raises(NotImplementedError):
        solver.load_model(tmp_path, format=fmt)


def test_load_model_rejects_unknown_format(tmp_path):
    solver = make_solver()
    with pytest.raises(ValueError, match="Unsupported format"):
        solver.load_model(tmp_path, format="xml")


def test_change_domain_to_real_converts_nonnegative_integers():
    solver = make_solver()
    var = pyo.Var(domain=pyo.NonNegativeIntegers)
    var.construct()
    solver._change_domain_to_real(var)
    assert var.domain is pyo.NonNegativeReals


def test_change_domain_to_real_converts_integers():
    solver = make_solver()
    var = pyo.Var(domain=pyo.Integers)
    var.construct()
    solver._change_domain_to_real(var)
    assert var.domain is pyo.Reals


def test_change_domain_to_real_converts_nonpositive_integers():
    solver = make_solver()
    var = pyo.Var(domain=pyo.NonPositiveIntegers)
    var.construct()
    solver._change_domain_to_real(var)
    assert var.domain is pyo.NonPositiveReals


def test_change_domain_to_real_converts_binary():
    solver = make_solver()
    var = pyo.Var(domain=pyo.Binary)
    var.construct()
    solver._change_domain_to_real(var)
    assert var.domain is pyo.Reals

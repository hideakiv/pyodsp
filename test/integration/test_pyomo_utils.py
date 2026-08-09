import pandas as pd
import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.solver.pyomo_utils import (
    add_linear_terms_to_objective,
    update_linear_terms_in_objective,
    add_terms_to_objective,
    update_quad_terms_in_objective,
    negate_objective_sense,
    negate_saved_objective_csv,
)


def make_solver(sense=pyo.minimize):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(-10, 10))
    model.y = pyo.Var(domain=pyo.Reals, bounds=(-10, 10))
    model.obj = pyo.Objective(expr=model.x + 2 * model.y, sense=sense)
    return PyomoSolver(model, SolverConfig("appsi_highs"), [model.x, model.y])


def test_add_terms_to_objective_deactivates_original_and_adds_ones():
    solver = make_solver()
    add_terms_to_objective(solver, [solver.model.x, solver.model.y])

    assert solver.original_objective.active is False
    assert solver.model._mod_obj.active is True

    solver.model.x.set_value(3.0)
    solver.model.y.set_value(4.0)
    # original: x + 2y = 3 + 8 = 11 ; mod_obj adds 1*x + 1*y = 7 -> 18
    assert pyo.value(solver.model._mod_obj.expr) == 18.0


def test_add_linear_terms_to_objective_uses_given_coefficients():
    solver = make_solver()
    add_linear_terms_to_objective(
        solver, [10.0, 100.0], [solver.model.x, solver.model.y]
    )

    solver.model.x.set_value(1.0)
    solver.model.y.set_value(1.0)
    # original: x + 2y = 3 ; plus 10*x + 100*y = 110 -> 113
    assert pyo.value(solver.model._mod_obj.expr) == 113.0
    assert solver.model._mod_obj.sense == solver.original_objective.sense


def test_update_linear_terms_in_objective_replaces_previous_mod_obj():
    solver = make_solver()
    update_linear_terms_in_objective(solver, [1.0, 1.0], [solver.model.x, solver.model.y])
    first_mod_obj = solver.model._mod_obj

    update_linear_terms_in_objective(solver, [5.0, 5.0], [solver.model.x, solver.model.y])

    assert solver.model._mod_obj is not first_mod_obj
    solver.model.x.set_value(1.0)
    solver.model.y.set_value(1.0)
    # original: 1 + 2 = 3; plus 5*1 + 5*1 = 10 -> 13
    assert pyo.value(solver.model._mod_obj.expr) == 13.0


def test_update_quad_terms_in_objective_always_adds_the_penalty():
    # update_quad_terms_in_objective is only ever called on a `_mod_obj` that
    # CuttingPlaneMethod.build_theta_objective already built as minimize-sense
    # (regardless of the true optimization direction), so the penalty is
    # always added — there is no separate "maximize" behavior to test here.
    solver = make_solver(sense=pyo.minimize)
    add_terms_to_objective(solver, [solver.model.x, solver.model.y])

    update_quad_terms_in_objective(
        solver, [solver.model.x, solver.model.y], center=[0.0, 0.0], penalty=2.0
    )

    solver.model.x.set_value(1.0)
    solver.model.y.set_value(0.0)
    # mod_obj (linear, x=1,y=0): (x+2y) + (x+y) = 1 + 1 = 2
    # quad term: 0.5 * 2.0 * ((1-0)^2 + (0-0)^2) = 1.0 -> total 3.0
    assert pyo.value(solver.model._mod_quad_obj.expr) == 3.0
    assert solver.model._mod_quad_obj.sense == pyo.minimize


def test_update_quad_terms_in_objective_replaces_previous_mod_quad_obj():
    solver = make_solver()
    add_terms_to_objective(solver, [solver.model.x, solver.model.y])
    update_quad_terms_in_objective(
        solver, [solver.model.x, solver.model.y], center=[0.0, 0.0], penalty=1.0
    )
    first = solver.model._mod_quad_obj

    update_quad_terms_in_objective(
        solver, [solver.model.x, solver.model.y], center=[1.0, 1.0], penalty=1.0
    )

    assert solver.model._mod_quad_obj is not first


def test_negate_objective_sense_flips_maximize_to_minimize():
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.obj = pyo.Objective(expr=3 * model.x + 1, sense=pyo.maximize)

    negate_objective_sense(model)

    assert model.obj.sense == pyo.minimize
    model.x.set_value(2.0)
    assert pyo.value(model.obj.expr) == -7.0  # -(3*2+1)


def test_negate_objective_sense_flips_minimize_to_maximize():
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.obj = pyo.Objective(expr=3 * model.x + 1, sense=pyo.minimize)

    negate_objective_sense(model)

    assert model.obj.sense == pyo.maximize
    model.x.set_value(2.0)
    assert pyo.value(model.obj.expr) == -7.0


def test_negate_objective_sense_is_its_own_inverse():
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.obj = pyo.Objective(expr=3 * model.x + 1, sense=pyo.maximize)

    negate_objective_sense(model)
    negate_objective_sense(model)

    assert model.obj.sense == pyo.maximize
    model.x.set_value(2.0)
    assert pyo.value(model.obj.expr) == 7.0


def test_negate_objective_sense_raises_when_no_active_objective():
    model = pyo.ConcreteModel()
    model.x = pyo.Var()

    with pytest.raises(ValueError, match="No active objective"):
        negate_objective_sense(model)


def test_negate_saved_objective_csv_negates_bm_csv(tmp_path):
    pd.DataFrame({"obj_bound": [1.0, 2.0], "obj_val": [3.0, None]}).to_csv(
        tmp_path / "bm.csv"
    )

    negate_saved_objective_csv(tmp_path)

    df = pd.read_csv(tmp_path / "bm.csv", index_col=0)
    assert df["obj_bound"].tolist() == [-1.0, -2.0]
    assert df["obj_val"].tolist()[0] == -3.0
    assert pd.isna(df["obj_val"].tolist()[1])


def test_negate_saved_objective_csv_negates_pbm_csv(tmp_path):
    pd.DataFrame(
        {"obj_bound": [1.0], "center_val": [2.0], "obj_val": [3.0]}
    ).to_csv(tmp_path / "pbm.csv")

    negate_saved_objective_csv(tmp_path)

    df = pd.read_csv(tmp_path / "pbm.csv", index_col=0)
    assert df["obj_bound"].tolist() == [-1.0]
    assert df["center_val"].tolist() == [-2.0]
    assert df["obj_val"].tolist() == [-3.0]


def test_negate_saved_objective_csv_raises_when_neither_file_exists(tmp_path):
    with pytest.raises(FileNotFoundError):
        negate_saved_objective_csv(tmp_path)

import pyomo.environ as pyo

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig


def make_solver(sense=pyo.minimize, bounds=(0, 10)):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=bounds)
    model.obj = pyo.Objective(expr=model.x, sense=sense)
    return model, PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])


def test_solve_optimal_loads_solution():
    model, solver = make_solver(bounds=(3, 10))
    solver.solve()

    assert solver.is_optimal() is True
    assert solver.get_solution() == [3.0]
    assert solver.get_objective_value() == 3.0


def test_solve_infeasible_reports_status_and_leaves_solution_unset():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(0, 1))
    model.obj = pyo.Objective(expr=model.x)
    model.con = pyo.Constraint(expr=model.x >= 5)
    solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])

    solver.solve()

    assert solver.is_infeasible() is True
    assert solver.is_optimal() is False
    # documented behavior: load_solutions=False + only loading on is_optimal()
    # means the var keeps its pre-solve (unset) value after an infeasible solve.
    assert solver.get_solution() == [None]


def test_get_original_objective_value_after_optimal_solve():
    model, solver = make_solver(bounds=(2, 10))
    solver.solve()

    assert solver.get_original_objective_value() == 2.0


def test_get_dual_returns_constraint_duals():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(0, 10))
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.obj = pyo.Objective(expr=model.x)
    model.con = pyo.Constraint(expr=model.x >= 4)
    solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])

    solver.solve()
    duals = solver.get_dual([model.con])

    assert duals == [1.0]


def test_get_dual_ray_from_infeasible_model():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(0, 1))
    model.x.set_value(0.5)  # get_dual_ray fixes vars at their *current* point;
    # in production this is always set beforehand via var.fix(...) in the
    # caller (e.g. BdAlgLeafPyomo._fix_variables) before solve() is called.
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.obj = pyo.Objective(expr=model.x)
    model.con = pyo.Constraint(expr=model.x >= 5)
    solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])

    solver.solve()
    assert solver.is_infeasible() is True

    ray = solver.get_dual_ray([model.con])

    assert len(ray) == 1
    assert ray[0] != 0.0
    assert solver.get_infeasible_model_objective_value() > 0.0


def test_get_unbd_ray_from_unbounded_model():
    # mirrors dd/alg_leaf_pyomo.py: the original objective is a trivial 0
    # (deactivated in practice), and `_mod_obj` carries the real (dual-derived)
    # coefficients that get_unbd_ray() requires to already be on the model.
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(0, None))
    model.obj = pyo.Objective(expr=0.0)
    solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])
    from pyodsp.solver.pyomo_utils import update_linear_terms_in_objective

    solver.original_objective.deactivate()
    update_linear_terms_in_objective(solver, [-1.0], [model.x])
    solver.solve()
    assert solver.is_unbounded() is True

    ray = solver.get_unbd_ray()

    assert len(ray) == 1
    assert solver.get_unbounded_model_objective_value() is not None


def test_get_unbd_ray_raises_when_mod_obj_missing():
    import pytest

    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(0, None))
    model.obj = pyo.Objective(expr=-model.x)
    solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])
    solver.solve()
    assert solver.is_unbounded() is True

    with pytest.raises(ValueError, match="_mod_obj"):
        solver.get_unbd_ray()

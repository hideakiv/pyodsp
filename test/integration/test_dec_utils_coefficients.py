import pyomo.environ as pyo

from pyodsp.dec.utils import (
    get_nonzero_coefficients_from_model,
    get_nonzero_coefficients_from_constraint,
    get_nonzero_coefficients_group,
)


def make_model():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals)
    model.y = pyo.Var(domain=pyo.Reals)
    model.z = pyo.Var(domain=pyo.Reals)
    model.obj = pyo.Objective(expr=model.x + model.y + model.z)
    # only x and y appear here; z is coupled in a different constraint;
    # con_no_coupling has neither x nor y and should be skipped entirely.
    model.con_xy = pyo.Constraint(expr=2 * model.x + 3 * model.y <= 10)
    model.con_z = pyo.Constraint(expr=model.z <= 5)
    model.con_no_coupling = pyo.Constraint(expr=model.z >= -100)
    return model


def test_get_nonzero_coefficients_from_constraint_extracts_matching_vars():
    model = make_model()
    coupling = get_nonzero_coefficients_from_constraint(
        model.con_xy, [model.x, model.y]
    )

    assert coupling.coefficients == {0: 2.0, 1: 3.0}
    assert coupling.constraint is model.con_xy


def test_get_nonzero_coefficients_from_constraint_ignores_absent_vars():
    model = make_model()
    coupling = get_nonzero_coefficients_from_constraint(model.con_z, [model.x, model.y])

    assert coupling.coefficients == {}


def test_get_nonzero_coefficients_from_model_skips_constraints_without_coupling():
    model = make_model()
    coupling_list = get_nonzero_coefficients_from_model(model, [model.x, model.y])

    # only con_xy touches x/y; con_z and con_no_coupling touch neither
    assert len(coupling_list) == 1
    assert coupling_list[0].constraint is model.con_xy
    assert coupling_list[0].coefficients == {0: 2.0, 1: 3.0}


def test_get_nonzero_coefficients_from_model_handles_indexed_constraints():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals)
    model.obj = pyo.Objective(expr=model.x)
    model.idx = pyo.RangeSet(0, 1)

    def rule(m, i):
        return m.x * (i + 1) <= 10

    model.indexed_con = pyo.Constraint(model.idx, rule=rule)

    coupling_list = get_nonzero_coefficients_from_model(model, [model.x])

    assert len(coupling_list) == 2
    coeffs = sorted(c.coefficients[0] for c in coupling_list)
    assert coeffs == [1.0, 2.0]


def test_get_nonzero_coefficients_group_separates_by_group_key():
    model = make_model()
    vars_dict = {"a": [model.x, model.y], "b": [model.z]}

    lagrangian_data = get_nonzero_coefficients_group(model, vars_dict)

    assert lagrangian_data.vars_dict == vars_dict
    assert len(lagrangian_data.constraints) == 3
    # con_xy row: group "a" has both coefficients, group "b" is empty
    xy_index = lagrangian_data.constraints.index(model.con_xy)
    assert lagrangian_data.matrix["a"][xy_index] == {0: 2.0, 1: 3.0}
    assert lagrangian_data.matrix["b"][xy_index] == {}


def test_get_nonzero_coefficients_group_records_constraint_bounds():
    model = make_model()
    vars_dict = {"a": [model.x, model.y]}

    lagrangian_data = get_nonzero_coefficients_group(model, vars_dict)

    xy_index = lagrangian_data.constraints.index(model.con_xy)
    assert lagrangian_data.lbs[xy_index] is None
    assert lagrangian_data.ubs[xy_index] == 10

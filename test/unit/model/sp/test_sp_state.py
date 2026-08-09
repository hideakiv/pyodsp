import pyomo.environ as pyo
import pytest

from pyodsp.model.sp.state import (
    add_state_link,
    appears_in_objective,
    domain_and_bounds,
    flatten,
    integer_variables,
    relax_integer_variables,
    replicate_free,
    resolve_state_specs,
    state_labels,
)


def make_model():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 10))
    model.y = pyo.Var(["a", "b"], domain=pyo.Reals)
    return model


def test_state_defaults_to_every_first_stage_variable():
    specs = resolve_state_specs(make_model(), None)

    assert [spec.name for spec in specs] == ["x", "y"]
    assert state_labels(specs) == ["x", "y[a]", "y[b]"]


def test_named_state_keeps_the_order_given():
    specs = resolve_state_specs(make_model(), ["y", "x"])

    assert state_labels(specs) == ["y[a]", "y[b]", "x"]


def test_an_unknown_state_name_lists_what_exists():
    with pytest.raises(ValueError, match=r"No variable named 'z'.*\['x', 'y'\]"):
        resolve_state_specs(make_model(), ["z"])


def test_a_model_with_no_variables_is_rejected():
    with pytest.raises(ValueError, match="declares no variables"):
        resolve_state_specs(pyo.ConcreteModel(), None)


def test_flatten_follows_the_label_order():
    model = make_model()
    specs = resolve_state_specs(model, None)

    assert flatten(model, specs) == [model.x, model.y["a"], model.y["b"]]


def test_a_replica_is_read_in_the_reference_order():
    # The replica's index set is built from the captured order, so the two
    # coupling lists line up position by position — which is all pyodsp.dec
    # checks.
    reference = make_model()
    specs = resolve_state_specs(reference, None)

    replica = pyo.ConcreteModel()
    replicate_free(replica, specs)

    assert [v.name for v in flatten(replica, specs)] == [
        v.name for v in flatten(reference, specs)
    ]


def test_replicas_are_free_so_any_trial_point_can_be_fixed():
    reference = make_model()
    specs = resolve_state_specs(reference, None)

    replica = pyo.ConcreteModel()
    replicate_free(replica, specs)

    for var in flatten(replica, specs):
        assert var.domain is pyo.Reals
        assert var.bounds == (None, None)


def test_replicating_onto_a_name_already_in_use_is_rejected():
    reference = make_model()
    specs = resolve_state_specs(reference, ["x"])

    replica = pyo.ConcreteModel()
    replica.x = pyo.Var()

    with pytest.raises(ValueError, match="already has a component named 'x'"):
        replicate_free(replica, specs)


def test_state_view_exposes_the_replica_and_refuses_anything_else():
    reference = make_model()
    specs = resolve_state_specs(reference, None)
    replica = pyo.ConcreteModel()
    view = replicate_free(replica, specs)

    assert view.x is replica.x
    assert view["y"] is replica.y
    assert list(view) == ["x", "y"]

    with pytest.raises(AttributeError, match="not a state variable"):
        view.nope

    with pytest.raises(AttributeError, match="read-only"):
        view.x = 1


def test_integer_variables_are_found_and_excludable():
    model = pyo.ConcreteModel()
    model.a = pyo.Var(domain=pyo.Binary)
    model.b = pyo.Var(domain=pyo.NonNegativeIntegers)
    model.c = pyo.Var(domain=pyo.Reals)

    assert sorted(integer_variables(model)) == ["a", "b"]
    assert integer_variables(model, exclude=[model.a]) == ["b"]


def test_relaxing_keeps_the_bounds_that_make_it_a_relaxation():
    model = pyo.ConcreteModel()
    model.a = pyo.Var(domain=pyo.Binary)
    model.b = pyo.Var(domain=pyo.Integers, bounds=(2, 8))

    relaxed = relax_integer_variables(model)

    assert sorted(relaxed) == ["a", "b"]
    assert model.a.domain is pyo.Reals and model.a.bounds == (0, 1)
    assert model.b.domain is pyo.Reals and model.b.bounds == (2, 8)
    assert integer_variables(model) == []


def test_appears_in_objective_distinguishes_objective_from_constraint():
    model = pyo.ConcreteModel()
    model.s = pyo.Var()
    model.y = pyo.Var()
    model.con = pyo.Constraint(expr=model.s + model.y >= 1)
    model.obj = pyo.Objective(expr=model.y, sense=pyo.minimize)

    assert not appears_in_objective(model, [model.s])

    model.del_component(model.obj)
    model.obj = pyo.Objective(expr=model.y + 3 * model.s, sense=pyo.minimize)

    assert appears_in_objective(model, [model.s])


def test_appears_in_objective_sees_nonlinear_use():
    model = pyo.ConcreteModel()
    model.s = pyo.Var(initialize=1.0)
    model.obj = pyo.Objective(expr=model.s**2, sense=pyo.minimize)

    assert appears_in_objective(model, [model.s])


def test_state_link_gives_every_state_a_defining_constraint():
    model = pyo.ConcreteModel()
    model.s = pyo.Var([0, 1])
    model.obj = pyo.Objective(expr=model.s[0] + model.s[1], sense=pyo.minimize)
    state_vars = [model.s[0], model.s[1]]

    coupling = add_state_link(model, state_vars)

    assert len(coupling) == 2
    assert not appears_in_objective(model, coupling)
    assert len(model._sp_state_link) == 2


def test_domain_and_bounds_are_read_per_flat_position():
    model = pyo.ConcreteModel()
    model.a = pyo.Var(domain=pyo.Binary)
    model.b = pyo.Var(domain=pyo.NonNegativeReals, bounds=(1, 4))
    specs = resolve_state_specs(model, None)

    info = domain_and_bounds(specs, model)

    assert info[0] == (pyo.Binary, (0, 1))
    assert info[1] == (pyo.NonNegativeReals, (1, 4))

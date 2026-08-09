"""Build-level behaviour of the two-stage pipeline: which algorithm it
picks, what it warns about, and whether the two coupling lists line up.

Nothing here runs an algorithm; auto_bound is off throughout so building
stays solver-free.
"""

import logging
import warnings

import pyomo.environ as pyo
import pytest

from pyodsp.model.sp import StochasticProgram

DEMAND = {"lo": 3.0, "hi": 9.0}


def make(
    *,
    integer_recourse_vars: bool = False,
    integer_first_stage: bool = False,
    integer_only_in: str | None = None,
    scenarios=None,
    **kwargs,
) -> StochasticProgram:
    kwargs.setdefault("log_level", logging.CRITICAL)
    sp = StochasticProgram("t", auto_bound=False, **kwargs)

    @sp.first_stage
    def first_stage(m):
        m.x = pyo.Var(
            bounds=(0, 20),
            domain=pyo.NonNegativeIntegers
            if integer_first_stage
            else pyo.NonNegativeReals,
        )
        return 2.0 * m.x

    @sp.recourse
    def recourse(m, state, scenario):
        integral = integer_recourse_vars or scenario.name == integer_only_in
        m.short = pyo.Var(
            bounds=(0, 20),
            domain=pyo.NonNegativeIntegers if integral else pyo.NonNegativeReals,
        )
        m.meet = pyo.Constraint(expr=state.x + m.short >= scenario["demand"])
        return 7.0 * m.short

    sp.set_scenarios(
        scenarios
        if scenarios is not None
        else [{"name": n, "demand": d} for n, d in DEMAND.items()]
    )
    return sp


# -- algorithm selection ---------------------------------------------------


def test_a_continuous_problem_stays_on_benders():
    sp = make()
    sp.build()

    assert sp.resolved_method == "bd"


def test_an_integer_recourse_moves_benders_to_scaled_cuts():
    sp = make(integer_recourse_vars=True)

    with pytest.warns(UserWarning, match="Switching to Benders with scaled cuts"):
        sp.build()

    assert sp.resolved_method == "bdsc"


def test_an_integer_recourse_can_be_relaxed_instead():
    sp = make(integer_recourse_vars=True, integer_recourse="relax")

    with pytest.warns(UserWarning, match="Relaxing the integrality"):
        built = sp.build()

    assert sp.resolved_method == "bd"
    assert built.relaxed_variables == {"lo": ["short"], "hi": ["short"]}


def test_an_integer_first_stage_alone_leaves_benders_untouched():
    # The master is a MIP and Benders is fine with that; only the recourse
    # needs LP duals.
    sp = make(integer_first_stage=True)
    sp.build()

    assert sp.resolved_method == "bd"


def test_dual_decomposition_warns_about_its_duality_gap():
    sp = make(method="dd", integer_recourse_vars=True)

    with pytest.warns(UserWarning, match="Lagrangian dual"):
        sp.build()

    assert sp.resolved_method == "dd"


def test_dual_decomposition_is_quiet_on_a_convex_problem():
    # No integrality means no duality gap, so there is nothing to warn about.
    sp = make(method="dd")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.build()

    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    assert sp.resolved_method == "dd"


def test_dual_decomposition_needs_something_to_couple():
    sp = make(method="dd", scenarios=[{"name": "only", "demand": 5.0}])

    with pytest.raises(ValueError, match="at least two scenarios"):
        sp.build()


def test_a_scenario_that_alone_turns_integral_is_caught_not_mis_decomposed():
    # Selection probes the first scenario; this model only becomes integral
    # in the second, so the guard inside the Benders build is what catches
    # it rather than a silently invalid cut.
    sp = make(integer_only_in="hi")

    with pytest.raises(ValueError, match="still has integer recourse"):
        sp.build()


# -- coupling ---------------------------------------------------------------


def test_benders_couples_the_two_stages_position_by_position():
    sp = make()
    built = sp.build()

    master = [v.name for v in built.root_solver.get_vars()]
    assert master == built.labels
    for solver in built.leaf_solvers.values():
        assert [v.name for v in solver.get_vars()] == master


def test_the_state_vector_follows_the_declaration_order():
    sp = StochasticProgram("order", auto_bound=False, log_level=logging.CRITICAL)

    @sp.first_stage
    def first_stage(m):
        m.beta = pyo.Var([2, 1], domain=pyo.NonNegativeReals)
        m.alpha = pyo.Var(domain=pyo.NonNegativeReals)
        return m.alpha + sum(m.beta[i] for i in [1, 2])

    @sp.recourse
    def recourse(m, state, scenario):
        m.z = pyo.Var(domain=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=m.z >= state.alpha + state.beta[2])
        return m.z

    sp.set_scenarios([{"name": "a"}, {"name": "b"}])
    built = sp.build()

    # declaration order, and each Var's own index order — not sorted
    assert built.labels == ["beta[2]", "beta[1]", "alpha"]
    assert [v.name for v in built.root_solver.get_vars()] == built.labels


def test_only_the_named_state_is_coupled():
    sp = StochasticProgram("named", auto_bound=False, log_level=logging.CRITICAL)

    @sp.first_stage(state=["shared"])
    def first_stage(m):
        m.shared = pyo.Var(domain=pyo.NonNegativeReals)
        m.private = pyo.Var(domain=pyo.NonNegativeReals)
        m.link = pyo.Constraint(expr=m.private >= m.shared)
        return m.shared + m.private

    @sp.recourse
    def recourse(m, state, scenario):
        m.z = pyo.Var(domain=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=m.z >= state.shared)
        return m.z

    sp.set_scenarios([{"name": "a"}, {"name": "b"}])
    built = sp.build()

    assert built.labels == ["shared"]


def _partial_state_program(**kwargs):
    """A first stage with a variable deliberately left out of the state."""
    kwargs.setdefault("log_level", logging.CRITICAL)
    sp = StochasticProgram("partial", auto_bound=False, **kwargs)

    @sp.first_stage(state=["shared"])
    def first_stage(m):
        m.shared = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeReals)
        m.private = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeReals)
        m.link = pyo.Constraint(expr=m.private >= m.shared)
        return 2.0 * m.shared + m.private

    @sp.recourse
    def recourse(m, state, scenario):
        m.short = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeReals)
        m.meet = pyo.Constraint(expr=state.shared + m.short >= scenario["demand"])
        return 7.0 * m.short

    sp.set_scenarios([{"name": n, "demand": d} for n, d in DEMAND.items()])
    return sp


def test_benders_tolerates_a_first_stage_variable_outside_the_state():
    # Its master keeps the whole first stage, so an uncoupled first-stage
    # variable simply stays there and never needs to reach a scenario.
    sp = _partial_state_program()
    built = sp.build()

    assert sp.resolved_method == "bd"
    assert built.labels == ["shared"]


@pytest.mark.parametrize("method", ["bdsc", "dd"])
def test_the_embedding_algorithms_require_every_first_stage_variable(method):
    # Both put a copy of the first stage inside every scenario, so a
    # variable outside the state vector gets an independent value per
    # scenario with nothing holding them equal — it silently stops being a
    # here-and-now decision.
    sp = _partial_state_program(method=method)

    with pytest.raises(ValueError, match=r"needs every first-stage variable"):
        sp.build()


def test_the_message_names_the_variables_and_the_way_out():
    sp = _partial_state_program(method="dd")

    with pytest.raises(ValueError) as excinfo:
        sp.build()

    message = str(excinfo.value)
    assert "'private'" in message
    assert "Drop the state=[...] argument" in message
    assert "method='bd'" in message


def test_an_integer_recourse_that_reroutes_to_bdsc_is_checked_too():
    # method='bd' with a partial state is fine until integrality sends it
    # to BDSC, which is not — the switch must not smuggle past the check.
    sp = StochasticProgram("partial_int", auto_bound=False, log_level=logging.CRITICAL)

    @sp.first_stage(state=["shared"])
    def first_stage(m):
        m.shared = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeReals)
        m.private = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeReals)
        m.link = pyo.Constraint(expr=m.private >= m.shared)
        return 2.0 * m.shared + m.private

    @sp.recourse
    def recourse(m, state, scenario):
        m.short = pyo.Var(bounds=(0, 20), domain=pyo.NonNegativeIntegers)
        m.meet = pyo.Constraint(expr=state.shared + m.short >= scenario["demand"])
        return 7.0 * m.short

    sp.set_scenarios([{"name": n, "demand": d} for n, d in DEMAND.items()])

    with pytest.warns(UserWarning, match="Switching to Benders with scaled cuts"):
        with pytest.raises(ValueError, match=r"needs every first-stage variable"):
            sp.build()


def test_dual_decomposition_replicates_the_state_once_per_scenario():
    sp = make(method="dd")
    built = sp.build()

    coupling = built.coupling_model
    assert len(coupling.state) == len(built.labels) * len(sp.scenarios)
    # non-anticipativity is what dual decomposition relaxes
    assert len(coupling.nonanticipativity) == len(built.labels) * len(sp.scenarios)
    for solver in built.leaf_solvers.values():
        assert len(solver.get_vars()) == len(built.labels)


def test_dual_decomposition_keeps_the_first_stage_domain_on_its_replicas():
    sp = make(method="dd", integer_first_stage=True)

    with pytest.warns(UserWarning):
        built = sp.build()

    assert built.coupling_model.state[0, 0].domain is pyo.NonNegativeIntegers


# -- sense ------------------------------------------------------------------


def test_a_maximize_program_reaches_the_algorithms_as_a_minimize_one():
    sp = make(sense="max")
    built = sp.build()

    assert built.root_solver.is_minimize()
    for solver in built.leaf_solvers.values():
        assert solver.is_minimize()


def test_an_unknown_sense_is_rejected():
    with pytest.raises(ValueError, match="sense must be"):
        StochasticProgram("t", sense="sideways")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"method": "magic"}, "method must be one of"),
        ({"integer_recourse": "ignore"}, "integer_recourse must be one of"),
    ],
)
def test_unknown_options_are_rejected_up_front(kwargs, match):
    with pytest.raises(ValueError, match=match):
        StochasticProgram("t", **kwargs)


# -- builder contract -------------------------------------------------------


def test_a_builder_that_returns_nothing_says_so():
    sp = StochasticProgram("t", auto_bound=False, log_level=logging.CRITICAL)

    @sp.first_stage
    def first_stage(m):
        m.x = pyo.Var()

    @sp.recourse
    def recourse(m, state, scenario):
        return state.x

    sp.set_scenarios([{"name": "a"}])

    with pytest.raises(ValueError, match="first-stage builder returned None"):
        sp.build()


def test_a_recourse_builder_that_returns_nothing_names_the_scenario():
    sp = StochasticProgram("t", auto_bound=False, log_level=logging.CRITICAL)

    @sp.first_stage
    def first_stage(m):
        m.x = pyo.Var(bounds=(0, 1))
        return m.x

    @sp.recourse
    def recourse(m, state, scenario):
        m.z = pyo.Var(bounds=(0, 1))
        m.c = pyo.Constraint(expr=m.z >= state.x)

    sp.set_scenarios([{"name": "only"}])

    with pytest.raises(ValueError, match=r"recourse builder \(scenario 'only'\)"):
        sp.build()


def test_declaring_an_objective_is_rejected_with_the_alternative():
    sp = StochasticProgram("t", auto_bound=False, log_level=logging.CRITICAL)

    @sp.first_stage
    def first_stage(m):
        m.x = pyo.Var(bounds=(0, 1))
        m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
        return m.x

    @sp.recourse
    def recourse(m, state, scenario):
        return state.x

    sp.set_scenarios([{"name": "a"}])

    with pytest.raises(ValueError, match="Return the objective expression instead"):
        sp.build()


def test_a_recourse_variable_colliding_with_a_state_name_is_rejected():
    sp = StochasticProgram("t", auto_bound=False, log_level=logging.CRITICAL)

    @sp.first_stage
    def first_stage(m):
        m.x = pyo.Var(bounds=(0, 1))
        return m.x

    @sp.recourse
    def recourse(m, state, scenario):
        m.x = pyo.Var()
        return m.x

    sp.set_scenarios([{"name": "a"}])

    with pytest.raises(ValueError, match="replaced the state variable"):
        sp.build()


def test_missing_declarations_are_reported():
    sp = StochasticProgram("t", auto_bound=False)
    with pytest.raises(ValueError, match="No first-stage builder"):
        sp.build()

    @sp.first_stage
    def first_stage(m):
        m.x = pyo.Var(bounds=(0, 1))
        return m.x

    with pytest.raises(ValueError, match="No recourse builder"):
        sp.build()


def test_missing_scenarios_are_reported():
    sp = StochasticProgram("t", auto_bound=False)

    @sp.first_stage
    def first_stage(m):
        m.x = pyo.Var(bounds=(0, 1))
        return m.x

    @sp.recourse
    def recourse(m, state, scenario):
        return state.x

    with pytest.raises(ValueError, match="No scenarios"):
        sp.build()


# -- scenario entry ---------------------------------------------------------


def test_add_scenario_defaults_to_equally_likely():
    sp = make()
    sp._scenarios = None
    sp._pending = []
    sp.add_scenario("a", demand=1.0).add_scenario("b", demand=2.0)

    assert sp.scenarios.probabilities == [0.5, 0.5]
    assert sp.scenarios[0].data == {"demand": 1.0}


def test_add_scenario_rejects_a_half_specified_set():
    sp = make()
    sp._scenarios = None
    sp._pending = []
    sp.add_scenario("a", demand=1.0, probability=0.5).add_scenario("b", demand=2.0)

    with pytest.raises(ValueError, match="Either every add_scenario call"):
        sp.scenarios


# -- introspection ----------------------------------------------------------


def test_describe_reports_the_algorithm_it_settled_on():
    sp = make(integer_recourse_vars=True)

    with pytest.warns(UserWarning):
        text = sp.describe()

    assert "bdsc" in text
    assert "asked for bd" in text
    assert "scenarios       : 2" in text


# -- the bypass -------------------------------------------------------------


def test_validate_false_lets_a_partial_state_through_to_bdsc():
    # The requirement still holds; it simply stops being enforced.
    sp = _partial_state_program(method="bdsc", validate=False)

    built = sp.build()

    assert sp.resolved_method == "bdsc"
    assert built.labels == ["shared"]


def test_validate_false_skips_the_state_replacement_check():
    sp = StochasticProgram(
        "unchecked", auto_bound=False, validate=False, log_level=logging.CRITICAL
    )

    @sp.first_stage
    def first_stage(m):
        m.x = pyo.Var(bounds=(0, 1))
        return m.x

    @sp.recourse
    def recourse(m, state, scenario):
        m.x = pyo.Var(bounds=(0, 1))  # would normally be rejected
        return m.x

    sp.set_scenarios([{"name": "a"}])

    assert sp.build().method == "bd"


def test_validate_false_skips_the_dual_decomposition_warning():
    sp = make(method="dd", integer_recourse_vars=True, validate=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.build()

    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    assert sp.resolved_method == "dd"


def test_validate_false_does_not_disturb_a_clean_problem():
    checked = make().build()
    unchecked = make(validate=False).build()

    assert checked.labels == unchecked.labels
    assert checked.method == unchecked.method


def test_bd_still_adapts_to_an_integer_recourse_without_validation():
    # Choosing between bd and bdsc is not a check — it decides which
    # algorithm is correct — so it survives validate=False.
    sp = make(integer_recourse_vars=True, validate=False)

    with pytest.warns(UserWarning, match="Switching to Benders with scaled cuts"):
        sp.build()

    assert sp.resolved_method == "bdsc"


# -- the deterministic equivalent -------------------------------------------


def test_the_deterministic_equivalent_builds_one_model_and_no_nodes():
    sp = make(method="de")
    built = sp.build()

    assert sp.resolved_method == "de"
    assert built.nodes == []
    assert built.root_node is None
    assert built.root_solver is not None
    # one block per scenario, on the single model
    assert set(built.scenario_blocks) == {"lo", "hi"}
    assert set(built.recourse_exprs) == {"lo", "hi"}


def test_it_needs_no_state_vector_of_its_own():
    # One model means one copy of the here-and-now decision, so
    # non-anticipativity holds by construction and a partial state is not
    # the hazard it is for BDSC and DD.
    sp = _partial_state_program(method="de")

    built = sp.build()

    assert sp.resolved_method == "de"
    assert built.labels == ["shared"]


def test_it_never_reroutes_on_an_integer_recourse():
    # There are no LP duals to be missing, so integrality is the solver's
    # problem and the method is taken at face value.
    sp = make(method="de", integer_recourse_vars=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.build()

    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    assert sp.resolved_method == "de"


def test_the_recourse_reads_the_real_first_stage_variable():
    sp = make(method="de")
    built = sp.build()

    model = built.root_solver.model
    # the coupling list is the model's own variables, not replicas
    assert built.root_solver.get_vars() == [model.x]

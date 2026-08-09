"""The scenario structure, independent of any model."""

import pytest

from pyodsp.model.msp.lattice import (
    ScenarioLattice,
    as_lattice,
    independent,
    markov,
    stage_varying,
)
from pyodsp.model.scenario import Scenario

REALIZATIONS = [
    {"name": "low", "probability": 0.25, "demand": 10.0},
    {"name": "high", "probability": 0.75, "demand": 30.0},
]


# -- stage-wise independent -------------------------------------------------


def test_independent_puts_one_node_in_the_first_stage():
    # The present is not uncertain: stage 0 is decided before anything is
    # observed.
    lattice = independent(REALIZATIONS, 4)

    assert lattice.num_stages == 4
    assert lattice.stage_size(0) == 1
    assert [lattice.stage_size(s) for s in range(4)] == [1, 2, 2, 2]


def test_independent_transitions_ignore_where_you_came_from():
    lattice = independent(REALIZATIONS, 3)

    assert lattice.transitions_from(0, 0) == [0.25, 0.75]
    # every node of stage 1 leads to stage 2 by the same distribution
    assert lattice.transitions_from(1, 0) == [0.25, 0.75]
    assert lattice.transitions_from(1, 1) == [0.25, 0.75]


def test_nodes_carry_their_position_and_data():
    node = independent(REALIZATIONS, 3).nodes(1)[1]

    assert (node.stage, node.index, node.name) == (1, 1, "high")
    assert node.idx == "1-1"
    assert node["demand"] == 30.0
    assert node.demand == 30.0
    assert not node.is_first and not node.is_last


def test_the_final_stage_is_marked_as_such():
    lattice = independent(REALIZATIONS, 3)

    assert all(node.is_last for node in lattice.nodes(2))
    assert lattice.nodes(0)[0].is_first


def test_unknown_node_data_names_what_is_available():
    node = independent(REALIZATIONS, 2).nodes(1)[0]

    with pytest.raises(AttributeError, match="demand"):
        node.supply


def test_a_single_stage_is_refused_with_a_pointer_to_the_two_stage_api():
    with pytest.raises(ValueError, match="at least 2"):
        independent(REALIZATIONS, 1)


# -- Markov -----------------------------------------------------------------


def test_markov_transitions_depend_on_the_current_state():
    lattice = markov(
        REALIZATIONS,
        3,
        transition_matrix=[[0.9, 0.1], [0.4, 0.6]],
        initial_distribution=[0.5, 0.5],
    )

    assert lattice.transitions_from(0, 0) == [0.5, 0.5]
    assert lattice.transitions_from(1, 0) == [0.9, 0.1]
    assert lattice.transitions_from(1, 1) == [0.4, 0.6]


def test_markov_falls_back_to_the_realization_probabilities():
    lattice = markov(REALIZATIONS, 3, transition_matrix=[[0.9, 0.1], [0.4, 0.6]])

    assert lattice.transitions_from(0, 0) == [0.25, 0.75]


def test_a_transition_matrix_of_the_wrong_size_is_refused():
    with pytest.raises(ValueError, match="rows but there are 2"):
        markov(REALIZATIONS, 3, transition_matrix=[[1.0]])


# -- validation -------------------------------------------------------------


def make_stages():
    return [
        [Scenario("root", 1.0, {})],
        [Scenario("a", 0.5, {}), Scenario("b", 0.5, {})],
    ]


def test_rows_must_be_distributions():
    with pytest.raises(ValueError, match="sums to"):
        ScenarioLattice(make_stages(), [[[0.5, 0.2]]])


def test_negative_probabilities_are_refused():
    with pytest.raises(ValueError, match="negative probability"):
        ScenarioLattice(make_stages(), [[[1.5, -0.5]]])


def test_row_width_must_match_the_next_stage():
    with pytest.raises(ValueError, match="entries but stage 1 has 2 nodes"):
        ScenarioLattice(make_stages(), [[[1.0]]])


def test_one_matrix_per_transition_is_required():
    with pytest.raises(ValueError, match="last stage transitions to nothing"):
        ScenarioLattice(make_stages(), [])


def test_more_than_one_first_stage_node_is_refused():
    stages = [
        [Scenario("a", 0.5, {}), Scenario("b", 0.5, {})],
        [Scenario("c", 1.0, {})],
    ]

    with pytest.raises(ValueError, match="Stage 0 must hold exactly one"):
        ScenarioLattice(stages, [[[1.0], [1.0]]])


def test_duplicate_names_within_a_stage_are_refused():
    stages = [
        [Scenario("root", 1.0, {})],
        [Scenario("a", 0.5, {}), Scenario("a", 0.5, {})],
    ]

    with pytest.raises(ValueError, match="duplicate node names"):
        ScenarioLattice(stages, [[[0.5, 0.5]]])


# -- reading the lattice ----------------------------------------------------


def test_reaching_probability_accumulates_through_the_stages():
    lattice = markov(
        REALIZATIONS,
        3,
        transition_matrix=[[1.0, 0.0], [0.0, 1.0]],
        initial_distribution=[0.5, 0.5],
    )

    reaching = lattice.reaching_probability()

    assert reaching[0] == [1.0]
    assert reaching[1] == [0.5, 0.5]
    # an absorbing chain keeps the mass where it started
    assert reaching[2] == [0.5, 0.5]


def test_as_lattice_passes_a_lattice_through_and_builds_from_realizations():
    lattice = independent(REALIZATIONS, 3)

    assert as_lattice(lattice) is lattice
    assert as_lattice(REALIZATIONS, 3).num_stages == 3

    with pytest.raises(ValueError, match="num_stages is needed"):
        as_lattice(REALIZATIONS)


# -- uncertainty that changes with time -------------------------------------


def test_stage_varying_allows_a_different_distribution_each_stage():
    lattice = stage_varying(
        [
            [
                {"name": "lo", "probability": 0.5, "d": 10.0},
                {"name": "hi", "probability": 0.5, "d": 20.0},
            ],
            [
                {"name": "a", "probability": 0.2, "d": 5.0},
                {"name": "b", "probability": 0.3, "d": 15.0},
                {"name": "c", "probability": 0.5, "d": 40.0},
            ],
        ]
    )

    assert [lattice.stage_size(s) for s in range(3)] == [1, 2, 3]
    # each stage is drawn with its own probabilities
    assert lattice.transitions_from(0, 0) == [0.5, 0.5]
    assert lattice.transitions_from(1, 0) == [0.2, 0.3, 0.5]
    assert lattice.nodes(2)[2]["d"] == 40.0


def test_stage_varying_accepts_transitions_of_its_own():
    lattice = stage_varying(
        [
            [{"name": "lo", "probability": 0.5}, {"name": "hi", "probability": 0.5}],
            [{"name": "lo", "probability": 0.5}, {"name": "hi", "probability": 0.5}],
        ],
        transitions=[[[0.5, 0.5]], [[0.9, 0.1], [0.4, 0.6]]],
    )

    # time-varying *and* origin-dependent
    assert lattice.transitions_from(1, 0) == [0.9, 0.1]
    assert lattice.transitions_from(1, 1) == [0.4, 0.6]


def test_stage_varying_needs_at_least_one_stage():
    with pytest.raises(ValueError, match="per_stage is empty"):
        stage_varying([])


# -- first-stage data -------------------------------------------------------


def test_the_first_stage_carries_no_data_by_default():
    node = independent(REALIZATIONS, 3).nodes(0)[0]

    assert node.data == {}
    assert node.get("demand", 0.0) == 0.0


def test_reaching_for_absent_first_stage_data_explains_why_it_is_absent():
    node = independent(REALIZATIONS, 3).nodes(0)[0]

    with pytest.raises(KeyError, match="decided before anything is observed"):
        node["demand"]


@pytest.mark.parametrize(
    "build",
    [
        lambda: independent(REALIZATIONS, 3, first_stage_data={"demand": 1.0}),
        lambda: markov(
            REALIZATIONS,
            3,
            transition_matrix=[[0.9, 0.1], [0.4, 0.6]],
            first_stage_data={"demand": 1.0},
        ),
        lambda: stage_varying(
            [REALIZATIONS, REALIZATIONS], first_stage_data={"demand": 1.0}
        ),
    ],
    ids=["independent", "markov", "stage_varying"],
)
def test_first_stage_data_can_be_supplied(build):
    # The first stage observes nothing, but it may still have known values
    # the builder needs.
    assert build().nodes(0)[0]["demand"] == 1.0


# -- label spreading, shared with the convergence chart ---------------------


def test_labels_that_would_overprint_are_pushed_apart():
    from pyodsp.viz.style import spread_labels

    entries = [{"y": 10.0}, {"y": 10.0}]

    spread_labels(entries, span=100.0)

    assert entries[0]["y"] != entries[1]["y"]


def test_labels_far_enough_apart_are_left_alone():
    from pyodsp.viz.style import spread_labels

    entries = [{"y": 10.0}, {"y": 90.0}]

    spread_labels(entries, span=100.0)

    assert [e["y"] for e in entries] == [10.0, 90.0]

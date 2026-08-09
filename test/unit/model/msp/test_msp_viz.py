"""The scenario lattice diagram.

Drawn from the lattice alone, so nothing here builds or solves a model.
"""

import pytest

from pyodsp.model.msp.lattice import independent, markov, stage_varying
from pyodsp.model.msp.viz import (
    LABEL_NODE_LIMIT,
    MAX_LATTICE_NODES_PER_STAGE,
    MAX_LATTICE_STAGES,
    plot_scenario_lattice,
)

pytest.importorskip("matplotlib")

REALIZATIONS = [
    {"name": "low", "probability": 0.25, "demand": 10.0},
    {"name": "high", "probability": 0.75, "demand": 30.0},
]


def test_it_draws_a_lattice(tmp_path):
    path = plot_scenario_lattice(independent(REALIZATIONS, 4), tmp_path / "l.png")

    assert path.exists() and path.stat().st_size > 0


def test_the_title_names_the_program_when_it_is_given():
    figure = plot_scenario_lattice(independent(REALIZATIONS, 3), name="hydro")

    assert "hydro" in figure.axes[0].get_title(loc="left")


def test_it_stands_alone_without_a_name():
    figure = plot_scenario_lattice(independent(REALIZATIONS, 3))

    assert figure.axes[0].get_title(loc="left") == "scenario lattice"


def test_the_subtitle_counts_the_whole_lattice():
    lattice = stage_varying([REALIZATIONS, REALIZATIONS, REALIZATIONS])
    figure = plot_scenario_lattice(lattice)

    # 1 root + three stages of two
    subtitle = " ".join(t.get_text() for t in figure.axes[0].texts)
    assert "4 stages" in subtitle
    assert "7 nodes" in subtitle


def test_a_stage_varying_lattice_draws_its_differing_widths(tmp_path):
    lattice = stage_varying(
        [
            REALIZATIONS,
            [
                {"name": "a", "probability": 0.2},
                {"name": "b", "probability": 0.3},
                {"name": "c", "probability": 0.5},
            ],
        ]
    )

    path = plot_scenario_lattice(lattice, tmp_path / "l.png")

    assert path.exists()


def test_a_markov_lattice_draws(tmp_path):
    lattice = markov(REALIZATIONS, 4, transition_matrix=[[0.9, 0.1], [0.4, 0.6]])

    assert plot_scenario_lattice(lattice, tmp_path / "l.png").exists()


def test_a_transition_that_cannot_happen_draws_no_edge():
    # An absorbing chain has zero-probability transitions; drawing them
    # would imply a path the model does not have.
    absorbing = markov(
        REALIZATIONS,
        3,
        transition_matrix=[[1.0, 0.0], [0.0, 1.0]],
        initial_distribution=[0.5, 0.5],
    )
    dense = independent(REALIZATIONS, 3)

    assert len(plot_scenario_lattice(absorbing).axes[0].lines) < len(
        plot_scenario_lattice(dense).axes[0].lines
    )


def test_a_lattice_too_wide_to_read_says_what_it_left_out():
    wide = [{"name": f"n{i}", "probability": 1.0 / 40} for i in range(40)]
    figure = plot_scenario_lattice(stage_varying([wide, wide]))

    subtitle = " ".join(t.get_text() for t in figure.axes[0].texts)
    assert "not shown" in subtitle
    # the full count is still reported, so the diagram never reads as complete
    assert "81 nodes" in subtitle


def test_node_names_are_dropped_once_there_are_too_many_to_read():
    few = plot_scenario_lattice(independent(REALIZATIONS, 3))
    crowded = plot_scenario_lattice(
        stage_varying(
            [[{"name": f"n{i}", "probability": 1.0 / 10} for i in range(10)]] * 2
        )
    )

    assert LABEL_NODE_LIMIT < MAX_LATTICE_NODES_PER_STAGE
    assert len(few.axes[0].texts) > len(crowded.axes[0].texts)


def test_the_stage_cap_is_honoured():
    long_horizon = independent(REALIZATIONS, MAX_LATTICE_STAGES + 5)
    figure = plot_scenario_lattice(long_horizon)

    assert len(figure.axes[0].get_xticks()) == MAX_LATTICE_STAGES

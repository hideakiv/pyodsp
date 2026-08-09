"""The convergence chart is algorithm-agnostic.

It works from a saved trajectory rather than a solved model, so nothing
here builds or runs a decomposition.
"""

import pandas as pd
import pytest

from pyodsp.viz.convergence import (
    plot_convergence,
    plot_run_convergence,
    read_trajectory,
)

pytest.importorskip("matplotlib")


def write_trajectory(node_dir, filename="bm.csv"):
    node_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"obj_bound": [-10.0, -4.0, -2.0], "obj_val": [None, 3.0, -2.0]}
    ).to_csv(node_dir / filename)
    return node_dir


def history(rows=3):
    return pd.DataFrame(
        {
            "iteration": range(rows),
            "bound": [-10.0, -4.0, -2.0][:rows],
            "incumbent": [None, 3.0, -2.0][:rows],
        }
    )


# -- reading a saved trajectory ---------------------------------------------


def test_reads_the_bundle_methods_csv(tmp_path):
    frame = read_trajectory(write_trajectory(tmp_path / "node0"))

    assert list(frame.columns) == ["iteration", "bound", "incumbent"]
    assert list(frame["iteration"]) == [0, 1, 2]
    assert frame["bound"].iloc[-1] == -2.0


def test_reads_the_proximal_variants_csv(tmp_path):
    # ProximalBundleMethod and RestrictedBundleMethod write pbm.csv
    frame = read_trajectory(write_trajectory(tmp_path / "node0", "pbm.csv"))

    assert frame["bound"].iloc[-1] == -2.0


def test_a_node_with_no_trajectory_says_which_nodes_have_one(tmp_path):
    empty = tmp_path / "node1"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="Only a node with a master"):
        read_trajectory(empty)


# -- plotting ---------------------------------------------------------------


def test_plots_from_a_dataframe(tmp_path):
    path = plot_convergence(history(), tmp_path / "c.png", title="Run")

    assert path.exists() and path.stat().st_size > 0


def test_plots_straight_from_a_run_directory(tmp_path):
    node_dir = write_trajectory(tmp_path / "node0")

    path = plot_run_convergence(node_dir, tmp_path / "c.png")

    assert path.exists() and path.stat().st_size > 0


def test_returns_the_figure_when_given_no_path():
    figure = plot_convergence(history())

    assert figure is not None
    assert figure.axes


def test_an_empty_history_is_refused():
    empty = pd.DataFrame(columns=["iteration", "bound", "incumbent"])

    with pytest.raises(ValueError, match="nothing to plot"):
        plot_convergence(empty)


def test_a_series_that_never_reported_does_not_derail_the_labels(tmp_path):
    # The incumbent has no value until the first cut arrives; a run that
    # stops before then leaves the column entirely empty.
    frame = history()
    frame["incumbent"] = None

    path = plot_convergence(frame, tmp_path / "c.png")

    assert path.exists()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_both_themes_render(tmp_path, theme):
    path = plot_convergence(history(), tmp_path / f"{theme}.png", theme=theme)

    assert path.exists() and path.stat().st_size > 0


def test_an_unknown_theme_is_refused():
    with pytest.raises(ValueError, match="theme must be"):
        plot_convergence(history(), theme="neon")


# -- a series the run never produced ----------------------------------------


def test_a_series_with_no_values_is_left_out_of_the_legend(tmp_path):
    """SDDP records no incumbent — it estimates that side by simulation.

    A legend entry for a line that was never drawn claims data the run
    does not have.
    """
    frame = history()
    frame["incumbent"] = None

    figure = plot_convergence(frame)
    _, labels = figure.axes[0].get_legend_handles_labels()

    assert labels == ["Bound"]
    assert figure.axes[0].get_legend() is None


def test_both_series_present_keeps_the_legend():
    figure = plot_convergence(history())

    assert figure.axes[0].get_legend() is not None


# -- the simulated interval SDDP has instead of an incumbent ----------------


def simulation(rows=2):
    return pd.DataFrame(
        {
            "iteration": [9, 19][:rows],
            "bound": [-2.0, -2.0][:rows],
            "mean": [-1.6, -1.8][:rows],
            "lower": [-1.9, -1.95][:rows],
            "upper": [-1.3, -1.65][:rows],
            "sample_size": [100, 100][:rows],
            "confidence_level": [0.95, 0.95][:rows],
        }
    )


def test_the_interval_is_drawn_when_there_is_no_incumbent(tmp_path):
    """SDDP estimates the other side rather than computing it.

    Error bars at the iterations actually tested, not a line implying a
    value was known at every step.
    """
    frame = history()
    frame["incumbent"] = None

    figure = plot_convergence(frame, simulation=simulation())
    _, labels = figure.axes[0].get_legend_handles_labels()

    assert labels == ["Bound", "Simulated policy (95% CI)"]
    # two series again, so the legend comes back
    assert figure.axes[0].get_legend() is not None


def test_the_confidence_level_is_named_in_the_legend():
    frame = simulation()
    frame["confidence_level"] = 0.9

    figure = plot_convergence(history(), simulation=frame)
    _, labels = figure.axes[0].get_legend_handles_labels()

    assert "90% CI" in labels[-1]


def test_an_empty_simulation_is_the_same_as_none(tmp_path):
    frame = history()
    frame["incumbent"] = None
    empty = pd.DataFrame(columns=["iteration", "mean", "lower", "upper"])

    figure = plot_convergence(frame, simulation=empty)
    _, labels = figure.axes[0].get_legend_handles_labels()

    assert labels == ["Bound"]


def test_reading_a_simulation_a_run_never_wrote(tmp_path):
    from pyodsp.viz.convergence import read_simulation

    assert read_simulation(tmp_path) is None


def test_reading_a_simulation_a_run_did_write(tmp_path):
    from pyodsp.viz.convergence import read_simulation

    simulation().to_csv(tmp_path / "simulation.csv", index=False)

    frame = read_simulation(tmp_path)

    assert list(frame["iteration"]) == [9, 19]

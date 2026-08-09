"""The bound-against-incumbent trajectory of a bundle-method run.

Every algorithm in pyodsp drives a BundleMethod, and every one of them
records the same two series per iteration, so this chart is about the
master's progress rather than about any particular decomposition — BD,
BDSC, DD and SDDP all produce one.

It reads the trajectory, not a solved model, so it works from a
DataFrame in memory or from the `bm.csv`/`pbm.csv` a finished run left on
disk. Those files are already in the units the models were written in
(BundleMethod.save converts them), so nothing here adjusts signs.
"""

from pathlib import Path

import pandas as pd

from .style import (
    figure,
    finish,
    formatter,
    spread_labels,
    style_axes,
    theme_colors,
    titles,
)

SIMULATION_FILE = "simulation.csv"
SIMULATION_SAMPLES_FILE = "simulation_samples.csv"
BOUND_COLUMN = "obj_bound"
INCUMBENT_COLUMN = "obj_val"
TRAJECTORY_FILES = ("bm.csv", "pbm.csv")


def read_trajectory(node_dir: str | Path) -> pd.DataFrame:
    """The trajectory a node saved, as iteration/bound/incumbent.

    Accepts either the bm.csv a BundleMethod writes or the pbm.csv from
    its proximal and restricted variants.

    Args:
        node_dir: A `node<idx>` directory under a run's output directory.

    Raises:
        FileNotFoundError: If neither file is there.
    """
    node_dir = Path(node_dir)
    for filename in TRAJECTORY_FILES:
        path = node_dir / filename
        if path.exists():
            frame = pd.read_csv(path, index_col=0)
            return pd.DataFrame(
                {
                    "iteration": range(len(frame)),
                    "bound": frame[BOUND_COLUMN].to_numpy(),
                    "incumbent": frame[INCUMBENT_COLUMN].to_numpy(),
                }
            )
    raise FileNotFoundError(
        f"No {' or '.join(TRAJECTORY_FILES)} in {node_dir}. Only a node with a "
        "master saves one — a Benders leaf records solutions and timings but "
        "no trajectory."
    )


def read_simulation(output_dir: str | Path) -> pd.DataFrame | None:
    """The convergence tests a Lattice recorded, if there are any.

    Only SDDP produces these; None means the run was not one, or stopped
    before its first test.
    """
    path = Path(output_dir) / SIMULATION_FILE
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return frame if not frame.empty else None


def read_simulation_samples(output_dir: str | Path):
    """The individual draws behind a run's last convergence test.

    None when the run recorded none — it was not an SDDP run, or stopped
    before its first test.
    """
    path = Path(output_dir) / SIMULATION_SAMPLES_FILE
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return frame["objective"] if not frame.empty else None


def plot_convergence(
    history: pd.DataFrame,
    path=None,
    *,
    simulation: pd.DataFrame | None = None,
    title: str = "Convergence",
    subtitle: str | None = None,
    theme: str = "light",
):
    """Plot a bound trajectory, against whatever bounds the other side.

    A cutting-plane run records an incumbent every iteration, and the
    shaded band between the two lines is the optimality gap it is driving
    to zero.

    SDDP has no incumbent: its bound is deterministic but the other side
    is *estimated*, by simulating the policy every so often. What it has
    is a confidence interval, so it is drawn as one — error bars at the
    iterations actually tested, rather than a line implying a value was
    known at every step.

    Args:
        history: Columns `iteration`, `bound` and `incumbent`. Missing
            entries are allowed — the incumbent has none until the first
            cut arrives, and SDDP never has one.
        path: Where to write the PNG. The figure is returned instead when
            this is None.
        simulation: Columns `iteration`, `mean`, `lower` and `upper` — one
            row per convergence test. Drawn as error bars when given.
        title: Chart title.
        subtitle: Optional line under it.
        theme: 'light' or 'dark'.
    """
    colors = theme_colors(theme)
    if history.empty:
        raise ValueError(
            "No convergence history was saved for this run, so there is "
            "nothing to plot."
        )

    fig, axes = figure(colors)
    iterations = history["iteration"]
    bound = history["bound"]
    incumbent = history["incumbent"]

    valid = bound.notna() & incumbent.notna()
    if valid.any():
        axes.fill_between(
            iterations[valid],
            bound[valid],
            incumbent[valid],
            color=colors["series_1"],
            alpha=0.10,
            linewidth=0,
        )

    # A series with no values at all is left out entirely, legend and all.
    # SDDP is the case that matters: it approaches the optimum from one
    # side and estimates the other by simulation, so it records no
    # incumbent — and a legend entry for a line that was never drawn
    # claims data the run does not have.
    drawn = [
        (series, color, label)
        for series, color, label in (
            (bound, colors["series_1"], "Bound"),
            (incumbent, colors["series_2"], "Incumbent"),
        )
        if series.notna().any()
    ]

    axes.plot(iterations, bound, color=colors["series_1"], linewidth=2.0, label="Bound")

    has_simulation = simulation is not None and not simulation.empty
    if has_simulation:
        _draw_simulation(axes, colors, simulation)

    if incumbent.notna().any():
        axes.plot(
            iterations,
            incumbent,
            color=colors["series_2"],
            linewidth=2.0,
            label="Incumbent",
            marker="o",
            markersize=3.5,
            markevery=max(1, len(history) // 20),
        )

    # End-of-line values. Each entry carries its own label position and
    # text, so a series that never produced a value simply contributes
    # nothing rather than shifting the others.
    fmt = formatter(list(bound) + list(incumbent))
    ends = []
    for series, color, _ in drawn:
        final = series.dropna()
        if final.empty:
            continue
        position = final.index[-1]
        ends.append(
            {
                "x": iterations.iloc[position],
                "y": final.iloc[-1],
                "text": fmt(final.iloc[-1]),
                "color": color,
            }
        )

    # A converged run puts both lines on the same point.
    spread_labels(ends, axes.get_ylim()[1] - axes.get_ylim()[0])

    for end in ends:
        # Value only — the legend and the label's own colour carry the
        # identity, and a shorter label is one that fits.
        axes.annotate(
            end["text"],
            xy=(end["x"], end["y"]),
            xytext=(8, 0),
            textcoords="offset points",
            color=end["color"],
            fontsize=9,
            fontweight="medium",
            va="center",
        )

    axes.set_xlabel("Iteration", color=colors["secondary"], fontsize=9.5)
    axes.set_ylabel("Objective", color=colors["secondary"], fontsize=9.5)
    style_axes(axes, colors)
    if len(drawn) + int(has_simulation) > 1:
        # One series needs no legend: the axis label and the title name it,
        # and a one-entry box is furniture around a fact already stated.
        legend = axes.legend(
            frameon=False, fontsize=9, loc="best", labelcolor=colors["secondary"]
        )
        legend.set_zorder(5)
    titles(axes, colors, title, subtitle)
    # Room for the end-of-line labels.
    axes.margins(x=0.12)
    return finish(fig, path, colors)


def plot_run_convergence(
    node_dir: str | Path,
    path=None,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    theme: str = "light",
):
    """Plot straight from a finished run's saved trajectory.

        plot_run_convergence("output/bdsc/cs/node0", "convergence.png")

    Args:
        node_dir: A `node<idx>` directory under a run's output directory.
        path: Where to write the PNG; the figure is returned when None.
        title: Defaults to naming the node the trajectory came from.
        subtitle: Optional line under it.
        theme: 'light' or 'dark'.
    """
    node_dir = Path(node_dir)
    return plot_convergence(
        read_trajectory(node_dir),
        path,
        title=title or f"{node_dir.name} — convergence",
        subtitle=subtitle,
        theme=theme,
    )


def _draw_simulation(axes, colors, simulation: pd.DataFrame) -> None:
    """The simulated policy value, as the interval it actually is.

    Error bars rather than a line: the policy is only evaluated every so
    many iterations, and each evaluation is an estimate from a finite
    sample. Joining them would claim both a value between the tests and a
    precision the sample does not have.
    """
    iterations = simulation["iteration"]
    mean = simulation["mean"]
    spans = [mean - simulation["lower"], simulation["upper"] - mean]
    label = "Simulated policy"
    if "confidence_level" in simulation:
        level = float(simulation["confidence_level"].iloc[0])
        label += f" ({level:.0%} CI)"

    axes.errorbar(
        iterations,
        mean,
        yerr=spans,
        fmt="o",
        markersize=4.5,
        color=colors["series_2"],
        ecolor=colors["series_2"],
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        label=label,
        zorder=4,
    )

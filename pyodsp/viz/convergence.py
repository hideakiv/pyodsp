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

from .style import figure, finish, formatter, style_axes, theme_colors, titles

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


def plot_convergence(
    history: pd.DataFrame,
    path=None,
    *,
    title: str = "Convergence",
    subtitle: str | None = None,
    theme: str = "light",
):
    """Plot a bound/incumbent trajectory.

    The shaded band between the two lines is the optimality gap, which is
    what the algorithm is actually driving to zero.

    Args:
        history: Columns `iteration`, `bound` and `incumbent`. Missing
            entries are allowed — the incumbent has none until the first
            cut arrives.
        path: Where to write the PNG. The figure is returned instead when
            this is None.
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

    axes.plot(iterations, bound, color=colors["series_1"], linewidth=2.0, label="Bound")
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
    for series, color in (
        (bound, colors["series_1"]),
        (incumbent, colors["series_2"]),
    ):
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

    # A converged run puts both lines on the same point; nudge the two
    # labels apart rather than overprinting them.
    if len(ends) == 2:
        span = abs(axes.get_ylim()[1] - axes.get_ylim()[0]) or 1.0
        if abs(ends[0]["y"] - ends[1]["y"]) < 0.07 * span:
            middle = (ends[0]["y"] + ends[1]["y"]) / 2.0
            ends[0]["y"] = middle + 0.045 * span
            ends[1]["y"] = middle - 0.045 * span

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

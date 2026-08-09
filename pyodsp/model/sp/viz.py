"""Charts specific to a solved stochastic program.

The shared palette and helpers, and the convergence trajectory — which
is about the master's progress rather than about stochastic programming —
live in pyodsp.viz.

matplotlib is an optional dependency (`pip install pyodsp[viz]`); nothing
else in the package imports this module, so a pyodsp without matplotlib
keeps working and only these functions raise.
"""

from pathlib import Path
from typing import List

from pyodsp.viz.convergence import plot_convergence
from pyodsp.viz.style import figure, finish, formatter, style_axes, theme_colors, titles

# Past this many scenarios the per-scenario forms stop being readable, so
# they show the largest contributors and say so on the chart rather than
# truncating quietly.
MAX_SCENARIOS_SHOWN = 24


def plot_scenario_objectives(result, path=None, *, theme: str = "light"):
    """Each scenario's objective, against the probability-weighted mean."""
    colors = theme_colors(theme)
    outcomes = [s for s in result.scenarios if s.objective is not None]
    if not outcomes:
        raise ValueError("No scenario objectives were recovered from this run.")

    dropped = max(0, len(outcomes) - MAX_SCENARIOS_SHOWN)
    if dropped:
        outcomes = sorted(outcomes, key=lambda s: abs(s.objective), reverse=True)
        outcomes = outcomes[:MAX_SCENARIOS_SHOWN]

    names = [s.name for s in outcomes]
    values = [s.objective for s in outcomes]
    weights = [s.probability for s in outcomes]
    mean = sum(w * v for w, v in zip(weights, values)) / (sum(weights) or 1.0)

    fig, axes = figure(colors, size=(max(6.4, 0.62 * len(names) + 2.6), 4.2))
    positions = range(len(names))
    axes.bar(list(positions), values, color=colors["series_1"], width=0.55, linewidth=0)
    axes.set_xticks(list(positions), names, rotation=45 if len(names) > 6 else 0)
    for tick in axes.get_xticklabels():
        tick.set_ha("right" if len(names) > 6 else "center")

    fmt = formatter(values + [mean])
    axes.axhline(mean, color=colors["series_2"], linewidth=2.0, linestyle="--")
    axes.annotate(
        f"expected  {fmt(mean)}",
        xy=(1, mean),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 5),
        textcoords="offset points",
        color=colors["series_2"],
        fontsize=9,
        ha="right",
        va="bottom",
    )

    if any(value < 0 for value in values):
        axes.axhline(0, color=colors["axis"], linewidth=1.0)

    axes.set_ylabel("Objective", color=colors["secondary"], fontsize=9.5)
    style_axes(axes, colors)
    kind = (
        "probability-weighted share of the total"
        if result.method == "dd"
        else "recourse objective"
    )
    subtitle = f"{kind} by scenario"
    if dropped:
        subtitle += f" · {dropped} smaller scenarios not shown"
    titles(axes, colors, f"{result.name} — scenarios", subtitle)
    return finish(fig, path, colors)


def plot_scenario_tree(result, path=None, *, theme: str = "light"):
    """The decomposition itself: one first-stage node feeding the scenarios.

    Deliberately a diagram rather than a chart — it answers "what did the
    pipeline build?", which is the question a reader new to decomposition
    actually has.
    """
    colors = theme_colors(theme)
    outcomes = list(result.scenarios)
    dropped = max(0, len(outcomes) - MAX_SCENARIOS_SHOWN)
    outcomes = outcomes[:MAX_SCENARIOS_SHOWN]

    fig, axes = figure(colors, size=(7.0, max(3.0, 0.42 * len(outcomes) + 1.8)))
    count = len(outcomes)
    ys = [count - 1 - i for i in range(count)]
    centre = (count - 1) / 2.0
    fmt = formatter([o.objective for o in outcomes])

    for y, outcome in zip(ys, outcomes):
        axes.plot(
            [0.16, 0.62],
            [centre, y],
            color=colors["axis"],
            linewidth=1.2,
            zorder=1,
        )
        axes.annotate(
            f"p={outcome.probability:.3g}",
            xy=(0.39, (centre + y) / 2.0),
            color=colors["muted"],
            fontsize=8,
            ha="center",
            va="center",
            zorder=3,
            # the edge runs underneath, so the label needs its own ground
            bbox=dict(facecolor=colors["surface"], edgecolor="none", pad=1.5),
        )
        axes.scatter([0.62], [y], s=90, color=colors["series_2"], zorder=2)
        label = outcome.name
        if outcome.objective is not None:
            label += f"   {fmt(outcome.objective)}"
        axes.annotate(
            label,
            xy=(0.66, y),
            color=colors["text"],
            fontsize=9.5,
            va="center",
            zorder=3,
        )

    axes.scatter([0.16], [centre], s=160, color=colors["series_1"], zorder=2)
    axes.annotate(
        "first stage",
        xy=(0.16, centre),
        xytext=(0, 14),
        textcoords="offset points",
        color=colors["text"],
        fontsize=9.5,
        ha="center",
        zorder=3,
    )

    axes.set_xlim(0.02, 1.12)
    axes.set_ylim(-0.9, count - 0.1)
    axes.axis("off")
    subtitle = f"{result.method.upper()} · {len(result.scenarios)} scenarios"
    if dropped:
        subtitle += f" · {dropped} not shown"
    titles(axes, colors, f"{result.name} — decomposition", subtitle)
    return finish(fig, path, colors)


def plot_sp_convergence(result, path=None, *, theme: str = "light"):
    """The shared convergence chart, titled from the program it came from.

    The chart itself belongs to pyodsp.viz — every algorithm records the
    same trajectory. All this adds is what a stochastic program knows
    about it: its name, the algorithm that ran, and the final gap.
    """
    return plot_convergence(
        result.history,
        path,
        title=f"{result.name} — convergence",
        subtitle=f"{result.method.upper()} · "
        + ("maximize" if result.is_maximize else "minimize")
        + (f" · final gap {result.gap:.3%}" if result.gap is not None else ""),
        theme=theme,
    )


def plot_all(result, directory, *, theme: str = "light") -> List[Path]:
    """Write every chart that applies to this result.

    A chart with nothing to show is skipped rather than raising, so this
    stays usable on a run that stopped early.
    """
    directory = Path(directory)
    charts = {
        "convergence.png": plot_sp_convergence,
        "scenario_objectives.png": plot_scenario_objectives,
        "scenario_tree.png": plot_scenario_tree,
    }
    written: List[Path] = []
    for filename, plotter in charts.items():
        try:
            written.append(plotter(result, directory / filename, theme=theme))
        except ValueError:
            continue
    return written

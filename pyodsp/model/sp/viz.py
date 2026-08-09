"""Charts for a solved stochastic program.

matplotlib is an optional dependency (`pip install pyodsp[viz]`); nothing
else in the package imports this module, so a pyodsp without matplotlib
keeps working and only these functions raise.

The palette is a validated two-hue categorical pair plus the neutral ink
and hairline tokens, in light and dark variants. Both variants clear the
lightness band, chroma floor, colourblind separation, normal-vision floor
and 3:1 contrast against their own surface.
"""

from pathlib import Path
from typing import Dict, List, Sequence

MATPLOTLIB_HINT = (
    "Plotting needs matplotlib, which pyodsp does not require. "
    "Install it with `pip install matplotlib` (or `pip install pyodsp[viz]`)."
)

LIGHT_THEME: Dict[str, str] = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series_1": "#2a78d6",
    "series_2": "#eb6834",
}

DARK_THEME: Dict[str, str] = {
    "surface": "#1a1a19",
    "text": "#ffffff",
    "secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series_1": "#3987e5",
    "series_2": "#d95926",
}

# Past this many scenarios the per-scenario forms stop being readable, so
# they show the largest contributors and say so on the chart rather than
# truncating quietly.
MAX_SCENARIOS_SHOWN = 24
MAX_STATES_SHOWN = 30


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(MATPLOTLIB_HINT) from exc
    return plt


def _theme(name: str) -> Dict[str, str]:
    if name == "light":
        return LIGHT_THEME
    if name == "dark":
        return DARK_THEME
    raise ValueError(f"theme must be 'light' or 'dark', got {name!r}")


def _figure(theme: Dict[str, str], size=(8.0, 4.5)):
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=size)
    figure.patch.set_facecolor(theme["surface"])
    axes.set_facecolor(theme["surface"])
    return figure, axes


def _style_axes(axes, theme: Dict[str, str], *, grid_axis: str = "y") -> None:
    """Recede the frame so the data carries the chart."""
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(theme["axis"])
        axes.spines[side].set_linewidth(1.0)
    axes.tick_params(colors=theme["muted"], labelsize=9, length=0)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_color(theme["secondary"])
    if grid_axis != "none":
        axes.grid(axis=grid_axis, color=theme["grid"], linewidth=1.0, alpha=1.0)
        axes.set_axisbelow(True)


def _titles(axes, theme: Dict[str, str], title: str, subtitle: str | None) -> None:
    axes.set_title(
        title,
        color=theme["text"],
        fontsize=12,
        fontweight="semibold",
        loc="left",
        pad=24 if subtitle else 10,
    )
    if subtitle:
        axes.annotate(
            subtitle,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 7),
            textcoords="offset points",
            color=theme["secondary"],
            fontsize=9.5,
            va="bottom",
            ha="left",
        )


def _finish(figure, path, theme: Dict[str, str]):
    figure.tight_layout()
    if path is None:
        return figure
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, facecolor=theme["surface"])
    _pyplot().close(figure)
    return path


def _formatter(values: Sequence[float | None]):
    """One number format for the whole chart.

    Deciding per value gives a chart labelled "250", "170", "80.00" — the
    precision jumping around reads as noise. The scale of the largest
    value picks the format for all of them.
    """
    finite = [abs(v) for v in values if v is not None]
    scale = max(finite) if finite else 0.0
    if scale >= 1e6 or (0.0 < scale < 1e-2):
        return lambda v: f"{v:.3g}"
    decimals = 0 if scale >= 100 else 1 if scale >= 10 else 2
    return lambda v: f"{v:,.{decimals}f}"


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------


def plot_convergence(result, path=None, *, theme: str = "light"):
    """The master's bound against the incumbent, iteration by iteration.

    The shaded band between the two lines is the optimality gap, which is
    what the algorithm is actually driving to zero.
    """
    colors = _theme(theme)
    history = result.history
    if history.empty:
        raise ValueError(
            "No convergence history was saved for this run, so there is "
            "nothing to plot."
        )

    figure, axes = _figure(colors)
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
    fmt = _formatter(list(bound) + list(incumbent))
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
    _style_axes(axes, colors)
    legend = axes.legend(
        frameon=False, fontsize=9, loc="best", labelcolor=colors["secondary"]
    )
    legend.set_zorder(5)
    _titles(
        axes,
        colors,
        f"{result.name} — convergence",
        f"{result.method.upper()} · "
        + ("maximize" if result.is_maximize else "minimize")
        + (f" · final gap {result.gap:.3%}" if result.gap is not None else ""),
    )
    # Room for the end-of-line labels.
    axes.margins(x=0.12)
    return _finish(figure, path, colors)


def plot_first_stage(result, path=None, *, theme: str = "light"):
    """The here-and-now decision, one bar per state variable."""
    colors = _theme(theme)
    values = {k: v for k, v in result.first_stage_flat.items() if v is not None}
    if not values:
        raise ValueError("The run recovered no first-stage solution to plot.")

    items = sorted(values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    dropped = max(0, len(items) - MAX_STATES_SHOWN)
    items = items[:MAX_STATES_SHOWN]
    labels = [label for label, _ in items][::-1]
    heights = [value for _, value in items][::-1]

    figure, axes = _figure(colors, size=(8.0, max(2.6, 0.42 * len(labels) + 1.5)))
    positions = range(len(labels))
    axes.barh(
        list(positions),
        heights,
        color=colors["series_1"],
        height=0.55,
        linewidth=0,
    )
    axes.set_yticks(list(positions), labels)

    fmt = _formatter(heights)
    span = max(abs(min(heights)), abs(max(heights))) or 1.0
    for position, value in zip(positions, heights):
        offset = 0.012 * span * (1 if value >= 0 else -1)
        axes.annotate(
            fmt(value),
            xy=(value + offset, position),
            color=colors["secondary"],
            fontsize=9,
            va="center",
            ha="left" if value >= 0 else "right",
        )

    if any(value < 0 for value in heights):
        axes.axvline(0, color=colors["axis"], linewidth=1.0)

    _style_axes(axes, colors, grid_axis="x")
    subtitle = "first-stage decision"
    if dropped:
        subtitle += f" · {dropped} smaller variables not shown"
    _titles(axes, colors, f"{result.name} — first stage", subtitle)
    axes.margins(x=0.16)
    return _finish(figure, path, colors)


def plot_scenario_objectives(result, path=None, *, theme: str = "light"):
    """Each scenario's objective, against the probability-weighted mean."""
    colors = _theme(theme)
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

    figure, axes = _figure(colors, size=(max(6.4, 0.62 * len(names) + 2.6), 4.2))
    positions = range(len(names))
    axes.bar(list(positions), values, color=colors["series_1"], width=0.55, linewidth=0)
    axes.set_xticks(list(positions), names, rotation=45 if len(names) > 6 else 0)
    for tick in axes.get_xticklabels():
        tick.set_ha("right" if len(names) > 6 else "center")

    fmt = _formatter(values + [mean])
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
    _style_axes(axes, colors)
    kind = (
        "probability-weighted share of the total"
        if result.method == "dd"
        else "recourse objective"
    )
    subtitle = f"{kind} by scenario"
    if dropped:
        subtitle += f" · {dropped} smaller scenarios not shown"
    _titles(axes, colors, f"{result.name} — scenarios", subtitle)
    return _finish(figure, path, colors)


def plot_scenario_tree(result, path=None, *, theme: str = "light"):
    """The decomposition itself: one first-stage node feeding the scenarios.

    Deliberately a diagram rather than a chart — it answers "what did the
    pipeline build?", which is the question a reader new to decomposition
    actually has.
    """
    colors = _theme(theme)
    outcomes = list(result.scenarios)
    dropped = max(0, len(outcomes) - MAX_SCENARIOS_SHOWN)
    outcomes = outcomes[:MAX_SCENARIOS_SHOWN]

    figure, axes = _figure(colors, size=(7.0, max(3.0, 0.42 * len(outcomes) + 1.8)))
    count = len(outcomes)
    ys = [count - 1 - i for i in range(count)]
    centre = (count - 1) / 2.0
    fmt = _formatter([o.objective for o in outcomes])

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
    _titles(axes, colors, f"{result.name} — decomposition", subtitle)
    return _finish(figure, path, colors)


def plot_all(result, directory, *, theme: str = "light") -> List[Path]:
    """Write every chart that applies to this result.

    A chart with nothing to show is skipped rather than raising, so this
    stays usable on a run that stopped early.
    """
    directory = Path(directory)
    charts = {
        "convergence.png": plot_convergence,
        "first_stage.png": plot_first_stage,
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

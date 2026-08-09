"""Charts specific to a solved multistage program.

The convergence trajectory lives in pyodsp.viz — every algorithm records
one. What is particular to a multistage program is what its policy *does*
over the horizon, which is what these draw.
"""

from pathlib import Path
from typing import Mapping, Sequence

from pyodsp.viz.style import (
    figure,
    finish,
    formatter,
    spread_labels,
    style_axes,
    theme_colors,
    titles,
)

# Two hues is what the palette validates for; past that the paths stop
# being distinguishable anyway.
MAX_PATHS_SHOWN = 2

# Past these the lattice stops being a diagram and becomes a smudge.
MAX_LATTICE_STAGES = 12
MAX_LATTICE_NODES_PER_STAGE = 12
# Node names only fit while there are few enough to read.
LABEL_NODE_LIMIT = 8


def plot_state_trajectory(
    result,
    paths: Mapping[str, Sequence[str]],
    variable: str,
    path=None,
    *,
    theme: str = "light",
):
    """One state variable over the horizon, one line per scenario path.

    SDDP leaves a policy rather than a schedule, so what a run decided is
    only visible by walking a path through it. Drawing two contrasting
    paths together is what shows the policy reacting — the same rule
    producing different decisions because the scenarios differ.

    Args:
        result: A solved MspResult.
        paths: Named scenario paths, each a list of node ids as
            MspResult.simulate takes.
        variable: Which state variable to draw. Must be one of the
            program's state variables.
        path: Where to write the PNG; the figure is returned when None.
        theme: 'light' or 'dark'.
    """
    colors = theme_colors(theme)
    if variable not in result.labels:
        raise ValueError(
            f"{variable!r} is not a state variable; this program's state is "
            f"{result.labels}"
        )
    if not paths:
        raise ValueError("No paths to draw")

    position = result.labels.index(variable)
    dropped = max(0, len(paths) - MAX_PATHS_SHOWN)
    shown = list(paths.items())[:MAX_PATHS_SHOWN]

    series = {}
    for label, node_path in shown:
        trajectory = result.simulate(list(node_path))
        # The last stage passes nothing on, so its own value is read
        # instead of the state it hands to a successor it does not have.
        series[label] = [
            step.next_state[position]
            if step.next_state
            else step.solution.get(variable)
            for step in trajectory
        ]

    fig, axes = figure(colors, size=(7.6, 4.2))
    palette = [colors["series_1"], colors["series_2"]]
    every = list(series.values())
    fmt = formatter([v for values in every for v in values if v is not None])

    ends = []
    for (label, values), color in zip(series.items(), palette):
        axes.plot(
            range(len(values)),
            values,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=label,
        )
        final = [(i, v) for i, v in enumerate(values) if v is not None]
        if final:
            index, value = final[-1]
            ends.append({"x": index, "y": value, "text": fmt(value), "color": color})

    # Paths that end at the same value — a reservoir drained by both, say —
    # would otherwise stack their labels on one point.
    spread_labels(ends, axes.get_ylim()[1] - axes.get_ylim()[0])
    for end in ends:
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

    axes.set_xlabel("Stage", color=colors["secondary"], fontsize=9.5)
    axes.set_ylabel(variable, color=colors["secondary"], fontsize=9.5)
    axes.set_xticks(range(result.num_stages))
    style_axes(axes, colors)
    if len(series) > 1:
        legend = axes.legend(
            frameon=False, fontsize=9, loc="best", labelcolor=colors["secondary"]
        )
        legend.set_zorder(5)

    subtitle = f"{result.num_stages} stages · one line per scenario path"
    if dropped:
        subtitle += f" · {dropped} more not shown"
    titles(axes, colors, f"{result.name} — {variable}", subtitle)
    axes.margins(x=0.12)
    return finish(fig, path, colors)


def plot_all(result, paths: Mapping[str, Sequence[str]], directory, *, theme="light"):
    """The convergence chart, the lattice, and one trajectory per state."""
    directory = Path(directory)
    written = [result.plot_convergence(directory / "convergence.png", theme=theme)]
    if result.lattice is not None:
        written.append(
            plot_scenario_lattice(
                result.lattice,
                directory / "scenario_lattice.png",
                name=result.name,
                theme=theme,
            )
        )
    if result.simulation_samples is not None and len(result.simulation_samples):
        written.append(
            plot_objective_distribution(
                result, directory / "objective_distribution.png", theme=theme
            )
        )
    for variable in result.labels:
        written.append(
            plot_state_trajectory(
                result,
                paths,
                variable,
                directory / f"trajectory_{_slug(variable)}.png",
                theme=theme,
            )
        )
    return written


def _slug(label: str) -> str:
    return label.replace("[", "_").replace("]", "").replace(",", "_")


def plot_scenario_lattice(
    lattice, path=None, *, name: str | None = None, theme: str = "light"
):
    """The scenario structure itself: stages, realizations and transitions.

    Deliberately a diagram rather than a chart — like the two-stage
    decomposition tree, it answers "what did the pipeline build?".

    What makes it a lattice rather than a tree is that it *recombines*:
    every node of a stage leads to every node of the next, which is what
    keeps the problem linear in the horizon instead of exponential. That
    also makes the edges dense, so a probability is drawn as the weight
    of its edge rather than written on it — there are too many to label.
    Node size is the chance of standing there at all.

    Args:
        lattice: A ScenarioLattice.
        path: Where to write the PNG; the figure is returned when None.
        name: The program's name, for the title.
        theme: 'light' or 'dark'.
    """
    colors = theme_colors(theme)
    stages = min(lattice.num_stages, MAX_LATTICE_STAGES)
    dropped_stages = lattice.num_stages - stages

    sizes = [lattice.stage_size(s) for s in range(stages)]
    shown = [min(size, MAX_LATTICE_NODES_PER_STAGE) for size in sizes]
    dropped_nodes = sum(size - k for size, k in zip(sizes, shown))
    reaching = lattice.reaching_probability()

    fig, axes = figure(
        colors, size=(max(6.4, 1.5 * stages), max(3.2, 0.55 * max(shown) + 2.0))
    )

    def ys(count):
        """Node positions, centred so the lattice reads symmetrically."""
        return [(count - 1) / 2.0 - i for i in range(count)]

    # Edges first, so nodes sit on top of them.
    for stage in range(stages - 1):
        origins, targets = ys(shown[stage]), ys(shown[stage + 1])
        for i, y0 in enumerate(origins):
            probabilities = lattice.transitions_from(stage, i)
            for j, y1 in enumerate(targets):
                probability = probabilities[j]
                if probability <= 0.0:
                    continue
                axes.plot(
                    [stage, stage + 1],
                    [y0, y1],
                    color=colors["axis"],
                    linewidth=0.6 + 2.4 * probability,
                    alpha=0.25 + 0.65 * probability,
                    zorder=1,
                    solid_capstyle="round",
                )

    label_nodes = max(shown) <= LABEL_NODE_LIMIT
    for stage in range(stages):
        positions = ys(shown[stage])
        nodes = lattice.nodes(stage)[: shown[stage]]
        mass = reaching[stage] if stage < len(reaching) else [1.0] * shown[stage]
        for i, (y, node) in enumerate(zip(positions, nodes)):
            weight = mass[i] if i < len(mass) else 0.0
            axes.scatter(
                [stage],
                [y],
                s=60 + 340 * weight,
                color=colors["series_1"] if node.is_first else colors["series_2"],
                zorder=2,
            )
            if label_nodes:
                axes.annotate(
                    node.name,
                    xy=(stage, y),
                    xytext=(0, -14),
                    textcoords="offset points",
                    color=colors["secondary"],
                    fontsize=8,
                    ha="center",
                    va="top",
                    zorder=3,
                )

    axes.set_xticks(range(stages))
    axes.set_xlabel("Stage", color=colors["secondary"], fontsize=9.5)
    axes.set_yticks([])
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(colors["axis"])
    axes.tick_params(colors=colors["muted"], labelsize=9, length=0)
    for label in axes.get_xticklabels():
        label.set_color(colors["secondary"])

    total = sum(lattice.stage_size(s) for s in range(lattice.num_stages))
    subtitle = (
        f"{lattice.num_stages} stages · {total} nodes · edge weight is the "
        "transition probability, node size the chance of reaching it"
    )
    if dropped_stages or dropped_nodes:
        subtitle += f" · {dropped_stages} stages and {dropped_nodes} nodes not shown"
    heading = "scenario lattice" if name is None else f"{name} — scenario lattice"
    titles(axes, colors, heading, subtitle)
    axes.margins(x=0.08, y=0.22)
    return finish(fig, path, colors)


def plot_objective_distribution(result, path=None, *, bins=None, theme="light"):
    """What the policy actually costs, over the simulated scenario paths.

    The confidence interval on the convergence chart is a summary of this
    sample; the sample itself says what the summary cannot — whether the
    distribution is skewed, bimodal, or carries a tail that matters. A
    policy is chosen on its expectation, and this is what that expectation
    was taken over.

    The deterministic bound is drawn alongside. The distance between it
    and the sample mean is the same gap the convergence chart closes, seen
    against the spread it is being compared with.

    Args:
        result: A solved MspResult whose run recorded samples.
        path: Where to write the PNG; the figure is returned when None.
        bins: Histogram bins, as matplotlib takes them. Chosen by the
            Freedman-Diaconis rule when omitted.
        theme: 'light' or 'dark'.
    """
    colors = theme_colors(theme)
    samples = result.simulation_samples
    if samples is None or len(samples) == 0:
        raise ValueError(
            "This run recorded no simulation samples, so there is no "
            "distribution to plot. Only SDDP produces them, and only once "
            "it has tested convergence at least once."
        )

    values = list(samples)
    fig, axes = figure(colors, size=(7.6, 4.4))
    axes.hist(
        values,
        # "auto" rather than the Freedman-Diaconis rule alone, which on a
        # few hundred paths gives so few bins the shape disappears.
        bins=bins if bins is not None else "auto",
        color=colors["series_1"],
        edgecolor=colors["surface"],
        linewidth=0.8,
    )

    mean = sum(values) / len(values)
    fmt = formatter(values)
    row = result.simulation.iloc[-1] if result.simulation is not None else None

    if row is not None:
        axes.axvspan(
            row["lower"],
            row["upper"],
            color=colors["series_2"],
            alpha=0.12,
            linewidth=0,
            zorder=0,
        )
    # The two reference lines sit close together by construction — the gap
    # between them is what converged — so their labels are pointed away
    # from each other and stacked, rather than left to overprint.
    mean_on_left = result.bound is None or mean <= result.bound

    axes.axvline(mean, color=colors["series_2"], linewidth=2.0, linestyle="--")
    _reference_label(
        axes, colors["series_2"], mean, f"mean  {fmt(mean)}", mean_on_left, -12
    )

    if result.bound is not None:
        # A neutral reference mark, not a third series: the bound is a
        # different kind of quantity from the sample it is compared with.
        axes.axvline(result.bound, color=colors["text"], linewidth=1.4)
        _reference_label(
            axes,
            colors["text"],
            result.bound,
            f"bound  {fmt(result.bound)}",
            not mean_on_left,
            -28,
        )

    axes.set_xlabel("Objective", color=colors["secondary"], fontsize=9.5)
    axes.set_ylabel("Scenario paths", color=colors["secondary"], fontsize=9.5)
    style_axes(axes, colors)

    subtitle = f"{len(values)} simulated paths"
    if row is not None:
        subtitle += f" · shaded band is the {row['confidence_level']:.0%} CI"
    titles(axes, colors, f"{result.name} — objective distribution", subtitle)
    return finish(fig, path, colors)


def _reference_label(axes, color, x, text, point_left: bool, dy: float) -> None:
    """A label beside a vertical reference line, pointing away from it."""
    axes.annotate(
        text,
        xy=(x, 1),
        xycoords=("data", "axes fraction"),
        xytext=(-6 if point_left else 6, dy),
        textcoords="offset points",
        color=color,
        fontsize=9,
        ha="right" if point_left else "left",
        va="top",
    )

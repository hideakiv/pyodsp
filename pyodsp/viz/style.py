"""Shared chart styling.

matplotlib is an optional dependency (`pip install pyodsp[viz]`). Nothing
outside pyodsp.viz and pyodsp.model.sp.viz imports it, and both reach it
through `pyplot()` here, so a pyodsp installed without matplotlib keeps
working and only plotting raises.

The palette is a validated two-hue categorical pair plus the neutral ink
and hairline tokens, in light and dark variants. Both variants clear the
lightness band, chroma floor, colourblind separation, normal-vision floor
and 3:1 contrast against their own surface. The dark column is the same
two hues stepped for the dark surface, not an inversion of the light one.
"""

from pathlib import Path
from typing import Dict, Sequence

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


def pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(MATPLOTLIB_HINT) from exc
    return plt


def theme_colors(name: str) -> Dict[str, str]:
    if name == "light":
        return LIGHT_THEME
    if name == "dark":
        return DARK_THEME
    raise ValueError(f"theme must be 'light' or 'dark', got {name!r}")


def figure(theme: Dict[str, str], size=(8.0, 4.5)):
    plt = pyplot()
    fig, axes = plt.subplots(figsize=size)
    fig.patch.set_facecolor(theme["surface"])
    axes.set_facecolor(theme["surface"])
    return fig, axes


def style_axes(axes, theme: Dict[str, str], *, grid_axis: str = "y") -> None:
    """Recede the frame so the data carries the chart.

    Args:
        grid_axis: Which axis carries the gridlines — 'y' for vertical
            bars and lines, 'x' for horizontal bars, where a y-grid would
            just draw lines through the bars.
    """
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(theme["axis"])
        axes.spines[side].set_linewidth(1.0)
    axes.tick_params(colors=theme["muted"], labelsize=9, length=0)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_color(theme["secondary"])
    axes.grid(axis=grid_axis, color=theme["grid"], linewidth=1.0, alpha=1.0)
    axes.set_axisbelow(True)


def titles(axes, theme: Dict[str, str], title: str, subtitle: str | None) -> None:
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


def finish(fig, path, theme: Dict[str, str]):
    """Save to `path`, or hand the figure back when there is none."""
    fig.tight_layout()
    if path is None:
        return fig
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=theme["surface"])
    pyplot().close(fig)
    return path


def formatter(values: Sequence[float | None]):
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


def spread_labels(entries, span: float, *, threshold=0.07, offset=0.045):
    """Nudge two end-of-line labels apart when they would overprint.

    Series that converge — which is what a converged run looks like —
    land on the same point, and two labels drawn there are unreadable.
    Entries are dicts carrying at least a 'y'; they are adjusted in place
    and returned.
    """
    if len(entries) != 2:
        return entries
    span = abs(span) or 1.0
    if abs(entries[0]["y"] - entries[1]["y"]) < threshold * span:
        middle = (entries[0]["y"] + entries[1]["y"]) / 2.0
        entries[0]["y"] = middle + offset * span
        entries[1]["y"] = middle - offset * span
    return entries

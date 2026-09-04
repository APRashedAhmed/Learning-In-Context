"""Shared paper-figure style: theme, palette, sizes, and panel export.

Single source of truth for the paper's figure conventions. Every figure script
in ``figures/`` calls :func:`apply_style` in its style cell and exports each
panel with :func:`save_panel`.

Size discipline: panels are exported at FINAL physical size — ``figsize`` in
real inches from the size vocabulary below, fonts at final point size. When
composing the figure, place panels at 100% and never rescale; a panel that does
not fit gets its ``figsize`` changed here/in its script and re-exported.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from matplotlib.figure import Figure

# --- Size vocabulary (inches; single-column journal layout ~7" text width) ---
FULL_WIDTH = 7.0
HALF_WIDTH = 3.4
THIRD_WIDTH = 2.25
PANEL_SQUARE = (3.0, 3.0)  # default panel figsize inherited from the analysis notebooks
PANEL_TUNING = (2.5, 3.0)  # tuning-profile figsize inherited from the activity notebooks

# --- Fonts ---
# The paper's figure font is Liberation Sans (metric-compatible with Arial,
# SIL-OFL — freely redistributable), VENDORED in ``fonts/`` beside this module
# and registered AHEAD of Arial so rendering never depends on system fonts.
# Whichever machine composes the final figures needs Liberation Sans installed
# too, so the live SVG text keeps its metrics.
FONT_FAMILY = ["Liberation Sans", "Arial", "sans-serif"]
_FONTS_DIR = Path(__file__).resolve().parent / "fonts"

# --- Palette ---
# Condition hues shared across figures (seaborn "deep" anchors).
PALETTE_CONDITION = {
    "Low": sns.color_palette("deep")[0],
    "Medium": sns.color_palette("deep")[1],
    "High": sns.color_palette("deep")[2],
}
SHORTENED_CONDITIONS = {
    "Hazard Rate": "HZ",
    "Contingency": "CT",
}


def get_color_palette(columns, color_number_tup, linspace_range=(0.5, 1), linspace_offset=1):
    """Colormap-sampled palette dict (ported from hmdcpd.visualization).

    Shared by the figure scripts (fig5/fig6/fig7 previously carried divergent
    private copies). ``color_number_tup`` entries are ``(cmap_name, n)`` for
    ``n`` evenly spaced samples, or ``(cmap_name, (n, j))`` for the single
    ``j``-th of ``n`` samples.
    """
    import numpy as np

    color_list = []
    for _i, (color, number) in enumerate(color_number_tup):
        cmap = sns.color_palette(color, as_cmap=True)
        if isinstance(number, int):
            num = number + linspace_offset
        elif isinstance(number, tuple):
            num = number[0] + linspace_offset
        color_array = [cmap(x) for x in np.linspace(*linspace_range, num=num)]
        if isinstance(number, int):
            color_list += [color_array[_j] for _j in range(number)]
        elif isinstance(number, tuple):
            color_list += [color_array[number[1]]]
    return {col: color for col, color in zip(columns, color_list)}

# Repo root (this file lives at src/learning_in_context/visualization/).
_REPO_ROOT = Path(__file__).resolve().parents[3]
PANELS_DIR = _REPO_ROOT / "figures" / "panels"


def _register_fonts() -> None:
    """Register the vendored Liberation Sans faces.

    ``font_manager.fontManager.addfont`` registers each vendored ``.ttf`` for
    this process, so the primary family resolves identically on every machine
    regardless of what fonts the system ships.
    """
    for ttf in sorted(_FONTS_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(ttf))


def apply_style() -> None:
    """Apply the paper's shared plotting theme. Call once per figure script."""
    _register_fonts()
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            # Live text in SVG exports so a vector editor can edit labels
            # (matplotlib's default outlines text to paths).
            "svg.fonttype": "none",
            # TrueType (Type 42) so text stays live if PDF output is ever added.
            "pdf.fonttype": 42,
        }
    )


def save_panel(fig, fig_no: int | str, name: str) -> Path:
    """Export one panel as live-text SVG to ``figures/panels/fig<N>/<name>.svg``.

    Panel names are stable identifiers — the composed figure links to these
    paths, so never rename an existing output.

    Args:
        fig: The matplotlib figure holding exactly this panel.
        fig_no: Paper figure number (``3`` → ``fig3/``).
        name: Semantic panel name without extension (e.g. ``"cwc_straight"``).

    Returns:
        The written path.
    """
    out_dir = PANELS_DIR / f"fig{fig_no}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.svg"
    # Deterministic SVG output: pin the hash salt (matplotlib otherwise
    # randomizes per-run ids embedded in the SVG) and strip the embedded
    # export date, so repeated exports are byte-identical.
    plt.rcParams["svg.hashsalt"] = name
    fig.savefig(
        out_path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    return out_path


# --- Per-panel decoration toggles ------------------------------------------
# Panels are composed by hand in an external vector editor, so the author
# chooses per panel which "frame furniture" — legend, axis labels, tick
# labels — that panel draws. ``PanelDecor`` names those choices declaratively
# and ``apply_decor`` applies them to an already-drawn axes as a post-process
# step, generalizing the ``legend``/``xlabel``/``ylabel``/``title`` kwargs
# that ``cwc_plots.plot_cwc_swarm`` already carries.


class _Keep:
    """Unique sentinel: "leave whatever the renderer already set"."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return "KEEP"


KEEP = _Keep()


@dataclass(frozen=True)
class PanelDecor:
    """Which frame furniture a panel draws, applied by :func:`apply_decor`.

    For the label and legend fields the sentinel :data:`KEEP` means "leave what
    the renderer set", ``None`` (or ``""`` for labels) means "hide", and a
    string overrides the text. Tick-label fields are plain booleans. Every
    default is the identity choice, so ``PanelDecor()`` is a literal no-op —
    a panel that does not opt in re-exports byte-identical.
    """

    title: str | None | _Keep = KEEP  # KEEP=leave · None/""=hide · str=override
    xlabel: str | None | _Keep = KEEP
    ylabel: str | None | _Keep = KEEP
    xticklabels: bool = True  # False -> tick_params(labelbottom=False)
    yticklabels: bool = True  # False -> tick_params(labelleft=False)
    legend: bool | _Keep = KEEP  # KEEP=leave · False=strip existing legend

    # Convenience constructors named by what they SUPPRESS, never by grid
    # position — composition is manual, so a position name would assert a layout
    # this code cannot see or validate and would go stale silently.
    @classmethod
    def shared_y(cls, **over) -> "PanelDecor":
        """Panel shares a neighbour's y-axis: hide its y-label and y ticks."""
        return cls(ylabel=None, yticklabels=False, **over)

    @classmethod
    def shared_x(cls, **over) -> "PanelDecor":
        """Panel shares a neighbour's x-axis: hide its x-label and x ticks."""
        return cls(xlabel=None, xticklabels=False, **over)

    @classmethod
    def no_legend(cls, **over) -> "PanelDecor":
        """Panel drops its legend (e.g. the legend moves to its own panel)."""
        return cls(legend=False, **over)


def apply_decor(ax, decor: PanelDecor) -> None:
    """Apply a :class:`PanelDecor` spec to an already-drawn axes.

    A mutating method is called only for a field that diverges from its default,
    so ``PanelDecor()`` touches nothing and re-exports byte-identical. ``None``
    (or ``""``) on a label hides it; a string overrides it. ``legend=False``
    strips an existing legend; ``legend=True`` is an explicit no-op (this
    post-processor has no handles with which to draw one).
    """
    if decor.title is not KEEP:
        ax.set_title(decor.title or "")
    if decor.xlabel is not KEEP:
        ax.set_xlabel(decor.xlabel or "")
    if decor.ylabel is not KEEP:
        ax.set_ylabel(decor.ylabel or "")
    if decor.xticklabels is False:
        ax.tick_params(labelbottom=False)
    if decor.yticklabels is False:
        ax.tick_params(labelleft=False)
    if decor.legend is False:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()


def make_legend_panel(
    handles,
    labels,
    *,
    title: str | None = None,
    ncol: int = 1,
    figsize: tuple[float, float] = (1.6, 1.2),
    frameon: bool = True,
    **legend_kw,
) -> Figure:
    """Render a legend alone on its own figure, for export as its own panel.

    Grab ``handles, labels = ax.get_legend_handles_labels()`` from a rendered
    panel, build the legend figure here, then export it with :func:`save_panel`
    (which pins ``svg.hashsalt`` per name, so the legend panel is deterministic
    too). The returned figure has no axes — its only artist is the legend.
    """
    fig = plt.figure(figsize=figsize)
    fig.legend(
        handles,
        labels,
        title=title,
        ncol=ncol,
        frameon=frameon,
        loc="center",
        **legend_kw,
    )
    return fig

"""Shared paper-figure style: theme, palette, sizes, and panel export.

Single source of truth for the paper's figure conventions (figures/SPEC.md).
Every figure script in ``figures/`` calls :func:`apply_style` in its style cell
and exports each panel with :func:`save_panel`.

Size discipline (SPEC rule 3): panels are exported at FINAL physical size —
``figsize`` in real inches from the size vocabulary below, fonts at final point
size. In Illustrator, place panels at 100% and never rescale; a panel that
does not fit gets its ``figsize`` changed here/in its script and re-exported.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# --- Size vocabulary (inches; single-column journal layout ~7" text width) ---
FULL_WIDTH = 7.0
HALF_WIDTH = 3.4
THIRD_WIDTH = 2.25
PANEL_SQUARE = (3.0, 3.0)  # matches the DS/figure notebooks' default figsize
PANEL_TUNING = (2.5, 3.0)  # matches figsize_tuning in the activity notebooks

# --- Fonts ---
# SPEC operator ruling 2 (2026-08-28): the decks used Arial, but the shipped
# figure font is Liberation Sans (metric-compatible, SIL-OFL — freely
# redistributable), VENDORED in ``fonts/`` beside this module and registered
# AHEAD of Arial so rendering never depends on system fonts. Illustrator
# machines need Liberation Sans installed too.
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
    """Register the vendored Liberation Sans faces (SPEC ruling 2).

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
            # Live text in SVG exports so Illustrator can edit labels
            # (matplotlib's default outlines text to paths). SPEC rule 2.
            "svg.fonttype": "none",
            # TrueType (Type 42) if PDF output is ever added. SPEC rule 2.
            "pdf.fonttype": 42,
        }
    )


def save_panel(fig, fig_no: int | str, name: str) -> Path:
    """Export one panel as live-text SVG to ``figures/panels/fig<N>/<name>.svg``.

    Panel names are stable identifiers — Illustrator compositions link to these
    paths, so never rename an existing output (SPEC rule 5).

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

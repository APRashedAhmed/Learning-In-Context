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
# The paper decks use Arial. Vendoring Arial is an OPEN QUESTION (SPEC open
# question 2: licensing); until ruled, register it only if present on the
# system and fall back silently to the free metric-compatible Liberation Sans.
FONT_FAMILY = ["Arial", "Liberation Sans", "sans-serif"]

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

# Repo root (this file lives at src/learning_in_context/visualization/).
_REPO_ROOT = Path(__file__).resolve().parents[3]
PANELS_DIR = _REPO_ROOT / "outputs" / "panels"


def _register_fonts() -> None:
    """Register preferred fonts if present; silently keep fallbacks otherwise."""
    for family in FONT_FAMILY[:-1]:
        try:
            font_manager.findfont(
                family, fallback_to_default=False, rebuild_if_missing=False
            )
        except ValueError:
            continue


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
    """Export one panel as live-text SVG to ``outputs/panels/fig<N>/<name>.svg``.

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
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    return out_path

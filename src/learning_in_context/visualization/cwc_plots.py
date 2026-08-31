"""Confidence-weighted-choice (CWC) panels: raw swarm under a mean trace.

One renderer covers both CWC panel shapes in the paper:

* **Split mode** — an ordinal x-category (grayzone position) split by a
  two-level condition (hazard rate). Each level gets its own colour from a
  light-to-dark ramp and its own mean trace, and a legend names the levels.
* **Family mode** — the category *is* the family (bounce contingency): one
  colour, one mean trace connecting the categories, and no legend, since the
  x tick labels already name the levels.

Family mode is selected by omitting ``hue`` or by passing ``hue == x``.

Styling is the caller's: nothing here touches global rcParams, so a figure
script calls :func:`~learning_in_context.visualization.paper_style.apply_style`
once and every panel inherits the paper's theme, fonts, and sizes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import paper_style

# --- Colour grammar -------------------------------------------------------
# Panels are drawn from a single sequential colormap family, sampled over a
# linspace range whose upper end is deliberately past 1.0: the colormap clamps
# there, so the darkest sample is the family's full-strength colour rather than
# the washed-out midpoint an evenly spaced ramp would land on.
SPLIT_FAMILY = "Blues"
SPLIT_LINSPACE = (0.4, 1.2)
FAMILY = "Reds"
FAMILY_LINSPACE = (0.75, 1.2)

# --- Mark sizes -----------------------------------------------------------
# Raw observations sit under the mean trace: small and semi-transparent, so a
# dense swarm reads as a cloud and never competes with the summary marks.
SWARM_SIZE = 4
SWARM_ALPHA = 0.333

# Beyond this many observations in the busiest category a swarm can no longer
# lay its points out side by side, so the panel falls back to a jittered strip,
# which degrades into a cloud of the same width instead of a pile-up. A CWC
# panel plots one point per participant or per model sample per category, so
# the threshold sits well above an ordinary panel and catches only frames an
# order of magnitude denser (raw per-trial rows, say).
SWARM_MAX = 200


def _levels(df: pd.DataFrame, column: str) -> list:
    """Category levels of ``column`` in order of first appearance."""
    return list(pd.unique(df[column].dropna()))


def _ramp(family: str, labels: Sequence, linspace_range: tuple) -> dict:
    """Map ``labels`` onto a light-to-dark ramp sampled from ``family``."""
    return paper_style.get_color_palette(
        labels,
        ((family, len(labels)),),
        linspace_range=linspace_range,
    )


def _split_palette(
    palette: Any,
    labels: Sequence,
    linspace_range: tuple | None,
) -> dict:
    """Resolve the split-mode palette to a ``{level: colour}`` mapping.

    ``palette`` is a colormap family name (sampled into a ramp over the hue
    levels), an explicit mapping, or a sequence of colours paired with
    ``labels`` in order.
    """
    if palette is None or isinstance(palette, str):
        return _ramp(palette or SPLIT_FAMILY, labels, linspace_range or SPLIT_LINSPACE)
    if isinstance(palette, Mapping):
        return dict(palette)
    return {label: color for label, color in zip(labels, palette, strict=False)}


def _family_color(palette: Any, linspace_range: tuple | None) -> Any:
    """Resolve the family-mode palette to the single colour the panel uses."""
    if palette is None or isinstance(palette, str):
        ramp = _ramp(palette or FAMILY, ["family"], linspace_range or FAMILY_LINSPACE)
        return ramp["family"]
    if isinstance(palette, Mapping):
        return next(iter(palette.values()))
    if isinstance(palette, Sequence) and not isinstance(palette, str):
        return palette[0]
    return palette


def _draw_raw_points(df: pd.DataFrame, x: str, swarm_max: int, **kwargs) -> None:
    """Draw every raw observation, as a swarm or — when too dense — a strip."""
    busiest = df.groupby(x, observed=True).size().max() if len(df) else 0
    if busiest > swarm_max:
        sns.stripplot(data=df, x=x, jitter=0.25, **kwargs)
    else:
        sns.swarmplot(data=df, x=x, **kwargs)


def _title_with_participant_count(
    title: str,
    df: pd.DataFrame,
    participant_col: str | None,
) -> str:
    """Annotate ``title`` with the number of participants behind the panel.

    The count goes in parentheses after the first word naming participants
    ("Participant CWC" -> "Participant (24) CWC"), or at the end of the title
    when no word does.
    """
    if participant_col and participant_col in df.columns:
        count = df[participant_col].nunique()
    else:
        count = len(df)

    words = title.split(" ")
    for i, word in enumerate(words):
        if "participant" in word.lower():
            words[i] = f"{word} ({count})"
            return " ".join(words)
    return f"{title} ({count})".strip()


def plot_cwc_swarm(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = paper_style.PANEL_SQUARE,
    ylim: tuple[float, float] | None = (-1.05, 1.05),
    ylabel: str | None = "Confidence Weighted Choice",
    xlabel: str | None = None,
    title: str | None = None,
    participant_count_title: bool = False,
    participant_col: str | None = "Participant ID",
    palette: Any = None,
    palette_labels: Sequence | None = None,
    hue_order: Sequence | None = None,
    legend: bool = True,
    legend_title: str | None = None,
    size: float = SWARM_SIZE,
    alpha: float = SWARM_ALPHA,
    errorbar: Any = "se",
    linspace_range: tuple[float, float] | None = None,
    swarm_max: int = SWARM_MAX,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw one CWC panel: a swarm of raw observations under a mean trace.

    Args:
        df: Tidy frame with one row per observation. Never modified.
        x: Categorical column for the x-axis.
        y: Numeric column holding the CWC values.
        hue: Condition column splitting each x-category. ``None`` — or the
            same column as ``x`` — selects family mode: one colour, one mean
            trace, no legend.
        ax: Axes to draw into. A new figure is created when omitted; when
            given, ``figsize`` is ignored and the axes' figure is returned.
        figsize: Figure size in inches for the axes this call creates.
        ylim: y-limits, or ``None`` to leave the data-driven limits.
        ylabel: y-axis label. ``None`` leaves the axis unlabelled.
        xlabel: x-axis label. Defaults to the title-cased ``x`` column name;
            pass ``""`` to suppress it.
        title: Axes title, omitted when ``None``.
        participant_count_title: Annotate ``title`` with the participant count.
        participant_col: Column counted for that annotation; the row count is
            used when it is absent from ``df``.
        palette: Colormap family name (``"Blues"``), an explicit
            ``{level: colour}`` mapping, or a sequence of colours — not a bare
            RGB tuple. Defaults to the blue ramp in split mode and the red
            family in family mode.
        palette_labels: Split mode only: the levels the ramp is sampled for,
            in light-to-dark order. Defaults to ``hue_order``, then to the
            levels' order of appearance in ``df``.
        hue_order: Draw order of the hue levels; in family mode it orders the
            x-categories instead.
        legend: Draw the hue legend. Family mode never draws one.
        legend_title: Legend title, defaulting to the title-cased ``hue``.
        size: Swarm marker size in points.
        alpha: Swarm marker opacity.
        errorbar: Error-bar spec for the mean marks, passed to seaborn.
        linspace_range: Colormap sampling range overriding the family default.
        swarm_max: Observations per category above which the raw layer is
            drawn as a jittered strip instead of a swarm.

    Returns:
        The figure and axes holding the panel.
    """
    split = hue is not None and hue != x

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if split:
        levels = _levels(df, hue)
        ramp_labels = list(palette_labels or hue_order or levels)
        order = list(hue_order or palette_labels or levels)
        # Keyed by level, so the ramp order and the draw order stay independent.
        colors = _split_palette(palette, ramp_labels, linspace_range)
        _draw_raw_points(
            df,
            x,
            swarm_max,
            y=y,
            hue=hue,
            hue_order=order,
            palette=colors,
            size=size,
            alpha=alpha,
            legend=False,
            zorder=1,
            ax=ax,
        )
        sns.pointplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            hue_order=order,
            palette=colors,
            errorbar=errorbar,
            legend=legend,
            zorder=2,
            ax=ax,
        )
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels,
                title=legend_title if legend_title is not None else str(hue).title(),
                loc="best",
            )
    else:
        # One colour for the whole panel, and the mean marks are drawn without
        # a hue so seaborn connects them into a single trace across the
        # categories rather than one isolated point per category.
        order = list(hue_order) if hue_order is not None else None
        color = _family_color(palette, linspace_range)
        _draw_raw_points(
            df,
            x,
            swarm_max,
            y=y,
            order=order,
            color=color,
            size=size,
            alpha=alpha,
            legend=False,
            zorder=1,
            ax=ax,
        )
        sns.pointplot(
            data=df,
            x=x,
            y=y,
            order=order,
            color=color,
            errorbar=errorbar,
            zorder=2,
            ax=ax,
        )

    if title is not None:
        ax.set_title(
            _title_with_participant_count(title, df, participant_col)
            if participant_count_title
            else title
        )

    ax.set_xlabel(xlabel if xlabel is not None else str(x).title())
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, ax

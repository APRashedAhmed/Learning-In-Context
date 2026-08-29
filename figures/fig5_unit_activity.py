"""Figure 5 — critical-unit activity panels (marimo, dual-use).

Retrofit of ``figures/fig_hazard_rate_activity.py`` (+ contingency computations
ported from ``figures/fig_contingency_activity.py``) into the paper-figure
pipeline defined by ``figures/SPEC.md``:

* the shared theme comes from ``paper_style.apply_style`` (SPEC rule 2: live-text
  SVG), not an inline ``sns.set_theme`` block;
* tier-2 transformations are memoized in
  ``learning_in_context.visualization.transforms`` (SPEC rule 8) so styling
  iteration never re-pays transformation cost;
* transform cells are split from render cells;
* each panel is exported self-contained via ``save_panel`` — no panel letters,
  no suptitles, no composed grids (SPEC rule 1).

Twelve panels (page-1 scope of the fig5 deck; VERIFIER RULING in
``tests/test_fig5_panels.py``) — three activity blocks × {hidden, cell} ×
{timecourse, profile}:

    Block 1  Hazard Rate, color change      (deck C/D/E/F)
    Block 2  Contingency, color change      (deck C/D/E/F, lower pair)
    Block 3  Contingency, no color change   (deck G/H/I/J)

Dual use (SPEC rule 7): ``marimo edit figures/fig5_unit_activity.py`` for
interactive work; ``python figures/fig5_unit_activity.py`` runs every cell
top-to-bottom (via ``app.run()``) and lands the 12 SVGs under
``outputs/panels/fig5/``.

Note (deviation from literal source; deck-verified): the contingency-block
profile scatters in ``fig_contingency_activity.py`` reference
``dict_model_stat_units_all["hz"]``, which is *empty* in that file
(commented-out copy-paste leftover) — the literal cells would ``KeyError``,
and their step criterion (``(targets[..., -4:-2] == 1).any``) does not
reproduce the deck either. The recipe below was recovered numerically from
the deck page ("Fig 5 - Crit Unit Behavior-1.png"): the **"hz" exemplar
units restricted to the 6 contingency models** (``STAT_UNITS["hz_cont"]``),
step criterion ``targets[..., -2] == 1`` (bounce color change) for block 2
and bounce-without-change for block 3 — all six models' (step, decay)
points match the deck panels exactly (decay is criterion-independent, so
matching decays uniquely fingerprint the unit set). See
``transforms.activity_change_profile``.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib

    matplotlib.use("Agg")  # headless: never reach for a GUI backend

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from learning_in_context.visualization import paper_style
    from learning_in_context.visualization import transforms

    return np, pd, plt, sns, paper_style, transforms


@app.cell
def _(np, paper_style):
    # Style cell: applies the shared theme AND returns every render-time constant.
    # Render cells consume these, which forces style-before-render in marimo's
    # dependency DAG (so svg.fonttype='none' is set before any panel is saved).
    paper_style.apply_style()

    def get_color_palette(columns, color_number_tup, linspace_range=(0.5, 1), linspace_offset=1):
        # Ported from hmdcpd.visualization.get_color_palette.
        color_list = []
        for _i, (color, number) in enumerate(color_number_tup):
            import seaborn as _sns

            cmap = _sns.color_palette(color, as_cmap=True)
            num = number + linspace_offset
            color_array = [cmap(x) for x in np.linspace(*linspace_range, num=num)]
            color_list += [color_array[_j] for _j in range(number)]
        return {col: color for col, color in zip(columns, color_list)}

    def viridis_palette(prefix, n):
        labels = [f"{prefix} {i + 1}" for i in range(n)]
        return get_color_palette(labels, (("viridis", n),), linspace_range=np.array((0.0, 1.1)))

    # Pinned per fig_hazard_rate_activity.py (dataset/model/exemplar model).
    DATASET = "extended_dataset"
    MODEL = "lstm"
    EXP_ID = "san-4604"
    T = 16
    CHANGE_IDX = 5
    M = 250
    COND_ORDER = ["Low", "High"]

    FIGSIZE_TC = paper_style.PANEL_TUNING  # time-courses
    FIGSIZE_PROFILE = paper_style.PANEL_SQUARE  # scatters

    return (
        DATASET,
        MODEL,
        EXP_ID,
        T,
        CHANGE_IDX,
        M,
        COND_ORDER,
        FIGSIZE_TC,
        FIGSIZE_PROFILE,
        viridis_palette,
    )


@app.cell
def _(plt, sns):
    # Render helpers (pure styling — SPEC rule 1: each panel self-contained).
    def render_timecourse(
        df,
        unit,
        unit_word,
        cond_order,
        change_labels,
        palette,
        change_idx,
        stat_short,
        vline_label,
        T,
        figsize,
        errorbar="se",
    ):
        num_conds = len(cond_order)
        fig, axes = plt.subplots(1, num_conds, figsize=figsize, sharey=True)
        if num_conds == 1:
            axes = [axes]
        for j, cond in enumerate(cond_order):
            ax = axes[j]
            ax.axvline(change_idx, linestyle="--", color="gray", label=vline_label)
            sub = df[
                (df["unit"] == unit)
                & (df["condition"] == cond)
                & (df["order"].isin(change_labels[cond]))
            ]
            sns.lineplot(
                data=sub,
                x="Timestep",
                y="Value",
                hue="order",
                palette=palette,
                errorbar=errorbar,
                seed=0,  # deterministic bootstrap CI bands across re-runs
                ax=ax,
                legend=(j == num_conds - 1),
            )
            ax.set_title(f"{cond} {stat_short} Trials")
            ax.set_xticks(range(0, T, 5))
            ax.set_ylabel(f"{unit_word} Unit Activity" if j == 0 else "")
        # One legend per panel, placed outside to the right (rule 1).
        last = axes[-1]
        if last.get_legend() is not None:
            sns.move_legend(
                last, "center left", bbox_to_anchor=(1.02, 0.5), frameon=True, title=None
            )
        fig.tight_layout()
        return fig

    def render_profile(df, state, figsize, point_size=100, alpha=0.7):
        x_col = f"{state}_step"
        y_col = f"{state}_decay"
        max_x = max(df[f"{s}_step"].max() for s in ["hidden", "cell"]) * 1.1
        max_y = max(df[f"{s}_decay"].max() for s in ["hidden", "cell"]) * 1.1

        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            ax=ax,
            s=point_size,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xlabel("Step Size")
        ax.set_ylabel("Activity Decay")
        ax.set_title(f"All {state.title()} Unit\nActivity Profiles")
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)
        fig.tight_layout()
        return fig

    return render_timecourse, render_profile


# ---------------------------------------------------------------------------
# Block 1 — Hazard Rate, color change
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""## Block 1 — Hazard Rate (color change)""")
    return


@app.cell
def _(CHANGE_IDX, DATASET, EXP_ID, M, MODEL, T, transforms):
    # Transform (memoized): hazard-rate ordered-change windows for the exemplar.
    df_hz_tc = transforms.ordered_change_windows(
        dataset=DATASET,
        model_name=MODEL,
        exp_id=EXP_ID,
        split_col="Hazard Rate",
        units=(15, 31),  # hidden 15, cell 31 (san-4604)
        T=T,
        k=1,
        change_idx=CHANGE_IDX,
        M=M,
        state_source="first_m_zscore",
        mask_mode="single_k",
        order_prefix="Change",
    )
    return (df_hz_tc,)


@app.cell
def _(DATASET, MODEL, transforms):
    df_hz_profile = transforms.activity_change_profile(
        dataset=DATASET,
        model_name=MODEL,
        criterion_mode="color_change",
        unit_set="hz",
    )
    return (df_hz_profile,)


@app.cell
def _(
    CHANGE_IDX,
    FIGSIZE_TC,
    T,
    df_hz_tc,
    paper_style,
    render_timecourse,
    viridis_palette,
):
    def _():
        change_labels = {
            "Low": [f"Change {i + 1}" for i in range(3)],
            "High": [f"Change {i + 1}" for i in range(8)],
        }
        palette = viridis_palette("Change", 8)
        for unit, word, name in (
            (15, "Hidden", "activity_timecourse_hazard_rate_hidden"),
            (31, "Cell", "activity_timecourse_hazard_rate_cell"),
        ):
            fig = render_timecourse(
                df_hz_tc,
                unit=unit,
                unit_word=word,
                cond_order=["Low", "High"],
                change_labels=change_labels,
                palette=palette,
                change_idx=CHANGE_IDX,
                stat_short="HZ",
                vline_label="Color Change",
                T=T,
                figsize=FIGSIZE_TC,
            )
            paper_style.save_panel(fig, 5, name)

    _()
    return


@app.cell
def _(FIGSIZE_PROFILE, df_hz_profile, paper_style, render_profile):
    def _():
        for state, name in (
            ("hidden", "activity_profile_hazard_rate_hidden"),
            ("cell", "activity_profile_hazard_rate_cell"),
        ):
            fig = render_profile(df_hz_profile, state, FIGSIZE_PROFILE)
            paper_style.save_panel(fig, 5, name)

    _()
    return


# ---------------------------------------------------------------------------
# Block 2 — Contingency, color change
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""## Block 2 — Contingency (color change)""")
    return


@app.cell
def _(CHANGE_IDX, DATASET, EXP_ID, M, MODEL, T, transforms):
    df_cont_tc = transforms.ordered_change_windows(
        dataset=DATASET,
        model_name=MODEL,
        exp_id=EXP_ID,
        split_col="Contingency",
        units=(11, 27),  # hidden 11, cell 27 (san-4604, "cont" exemplar units)
        T=T,
        k=2,
        change_idx=CHANGE_IDX,
        M=M,
        state_source="raw_states",
        mask_mode="single_k",
        order_prefix="Color Change",
    )
    return (df_cont_tc,)


@app.cell
def _(DATASET, MODEL, transforms):
    df_cont_profile = transforms.activity_change_profile(
        dataset=DATASET,
        model_name=MODEL,
        criterion_mode="bounce_color_change",
        unit_set="hz_cont",  # deck-verified: hz units of the 6 cont models
    )
    return (df_cont_profile,)


@app.cell
def _(
    CHANGE_IDX,
    FIGSIZE_TC,
    T,
    df_cont_tc,
    paper_style,
    render_timecourse,
    viridis_palette,
):
    def _():
        change_labels = {
            "Low": [f"Color Change {i + 1}" for i in range(3)],
            # Deck block-2 legend shows Color Change 1..6 (the literal source
            # cell currently says 7; the deck is the content anchor).
            "High": [f"Color Change {i + 1}" for i in range(6)],
        }
        palette = viridis_palette("Color Change", 6)
        for unit, word, name in (
            (11, "Hidden", "activity_timecourse_contingency_hidden"),
            (27, "Cell", "activity_timecourse_contingency_cell"),
        ):
            fig = render_timecourse(
                df_cont_tc,
                unit=unit,
                unit_word=word,
                cond_order=["Low", "High"],
                change_labels=change_labels,
                palette=palette,
                change_idx=CHANGE_IDX,
                stat_short=paper_style.SHORTENED_CONDITIONS["Contingency"],
                vline_label="Wall Bounce",
                T=T,
                figsize=FIGSIZE_TC,
                errorbar="ci",  # matches fig_contingency_activity.py render default
            )
            paper_style.save_panel(fig, 5, name)

    _()
    return


@app.cell
def _(FIGSIZE_PROFILE, df_cont_profile, paper_style, render_profile):
    def _():
        for state, name in (
            ("hidden", "activity_profile_contingency_hidden"),
            ("cell", "activity_profile_contingency_cell"),
        ):
            fig = render_profile(df_cont_profile, state, FIGSIZE_PROFILE)
            paper_style.save_panel(fig, 5, name)

    _()
    return


# ---------------------------------------------------------------------------
# Block 3 — Contingency, no color change
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""## Block 3 — Contingency (no color change)""")
    return


@app.cell
def _(CHANGE_IDX, DATASET, EXP_ID, M, MODEL, T, transforms):
    df_cont_nc_tc = transforms.ordered_change_windows(
        dataset=DATASET,
        model_name=MODEL,
        exp_id=EXP_ID,
        split_col="Contingency",
        units=(11, 27),
        T=T,
        k=2,
        change_idx=CHANGE_IDX,
        M=M,
        state_source="raw_states",
        mask_mode="bounce_no_change",
        order_prefix="No Color Change",
    )
    return (df_cont_nc_tc,)


@app.cell
def _(DATASET, MODEL, transforms):
    df_cont_nc_profile = transforms.activity_change_profile(
        dataset=DATASET,
        model_name=MODEL,
        criterion_mode="bounce_no_change",
        unit_set="hz_cont",  # deck-verified: hz units of the 6 cont models
    )
    return (df_cont_nc_profile,)


@app.cell
def _(
    CHANGE_IDX,
    FIGSIZE_TC,
    T,
    df_cont_nc_tc,
    paper_style,
    render_timecourse,
    viridis_palette,
):
    def _():
        change_labels = {
            "Low": [f"No Color Change {i + 1}" for i in range(7)],
            "High": [f"No Color Change {i + 1}" for i in range(7)],
        }
        palette = viridis_palette("No Color Change", 7)
        for unit, word, name in (
            (11, "Hidden", "activity_timecourse_contingency_no_change_hidden"),
            (27, "Cell", "activity_timecourse_contingency_no_change_cell"),
        ):
            fig = render_timecourse(
                df_cont_nc_tc,
                unit=unit,
                unit_word=word,
                cond_order=["Low", "High"],
                change_labels=change_labels,
                palette=palette,
                change_idx=CHANGE_IDX,
                stat_short=paper_style.SHORTENED_CONDITIONS["Contingency"],
                vline_label="Wall Bounce",
                T=T,
                figsize=FIGSIZE_TC,
                errorbar="ci",  # matches fig_contingency_activity.py render default
            )
            paper_style.save_panel(fig, 5, name)

    _()
    return


@app.cell
def _(FIGSIZE_PROFILE, df_cont_nc_profile, paper_style, render_profile):
    def _():
        for state, name in (
            ("hidden", "activity_profile_contingency_no_change_hidden"),
            ("cell", "activity_profile_contingency_no_change_cell"),
        ):
            fig = render_profile(df_cont_nc_profile, state, FIGSIZE_PROFILE)
            paper_style.save_panel(fig, 5, name)

    _()
    return


if __name__ == "__main__":
    app.run()

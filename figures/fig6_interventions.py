"""Figure 6 — critical-unit intervention panels (marimo, dual-use).

Ported from ``hmdcpd-analysis/notebooks/DS4-Interventions.py`` into the
paper-figure pipeline defined by ``figures/SPEC.md``:

* the shared theme comes from ``paper_style.apply_style`` (SPEC rule 2: live-text
  SVG), not an inline ``sns.set_theme`` block;
* the per-model intervention prediction frames are the memoized, shared tier-2
  transform ``transforms.intervention_prediction_frame`` (SPEC rule 8's
  "known-shared-from-day-one" — fig7's gate panels reuse it), so styling
  iteration never re-pays the load/window cost;
* transform cells are split from render cells;
* each deck box is exported as one self-contained SVG via ``save_panel`` — no
  panel letters, no suptitles, no composed grids (SPEC rule 1).

Eight panels (page-1 scope of the fig6 deck; contract in
``tests/test_fig6_panels.py``): two blocks × {hidden, cell} × {timecourse,
point plot}.

    Hazard-rate block   (deck G/H, I/J)  — straight trials, x 0-25
    Contingency block   (deck K/L, M/N)  — wall-bounce trials, x 0-7

Deviations from the literal DS4 source (deck-verified / ruled):

* Condition shorthand is uppercase ``HZ`` / ``CT`` from
  ``paper_style.SHORTENED_CONDITIONS`` (SPEC ruling 6), not DS4's mixed-case
  ``Hz`` / ``Cont`` — the single change that yields the contract's
  ``"Low to High HZ"`` / ``"High to Low CT"`` subplot titles.
* Panel N (``summary_pointplot_ct_cell``) title reads "All Models Cell Unit
  Interventions" — SPEC ruling 9 corrects the deck's/DS4's published
  copy-paste bug (they render "Hidden" over what is actually the cell-unit
  contingency frame ``dict_model_pred_dfs_melted_bounce_cont_c``). This is the
  one title that deliberately differs from the deck image.

Dual use (SPEC rule 7): ``marimo edit figures/fig6_interventions.py`` for
interactive work; ``python figures/fig6_interventions.py`` runs every cell
top-to-bottom (via ``app.run()``) and lands the 8 SVGs under
``outputs/panels/fig6/``.
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

    import math

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    from learning_in_context.visualization import paper_style
    from learning_in_context.visualization import transforms

    return (
        FuncFormatter,
        Line2D,
        math,
        np,
        pd,
        plt,
        sns,
        paper_style,
        transforms,
    )


@app.cell
def _(np, paper_style):
    # Style cell: applies the shared theme AND returns every render-time constant.
    # Render cells consume these, which forces style-before-render in marimo's
    # dependency DAG (so svg.fonttype='none' is set before any panel is saved).
    paper_style.apply_style()

    # Shared colormap-palette helper (promoted to paper_style; one copy).
    get_color_palette = paper_style.get_color_palette

    # Pipeline pinned per DS4-Interventions.py.
    MODEL = "lstm"
    EXP_ID = "san-4604"  # paper exemplar shown in the time-course panels (fig5)
    NUM_ALPHAS = 11
    ALPHAS = np.linspace(0, 1, NUM_ALPHAS)  # 0.0 … 1.0

    # DS4 windowing knobs.
    TIMESTEPS_PLOT = 26  # N for hazard-rate straight trials (x 0-25)
    N_CONT = TIMESTEPS_PLOT - 2  # N for contingency (x pre-offset)
    TIMESTEP_INTERVENTION = TIMESTEPS_PLOT - 24  # gray dashed vline at x=2
    BOUNCE_IDX = 15  # contingency wall-bounce onset (DS4 bounce_idx_2)

    # Final-timestep index the summary point plots collapse onto (DS4 magic
    # numbers): 24 for hazard-rate straight, 8 for the bounce-offset contingency.
    FINAL_T_HZ = 24
    FINAL_T_CT = 8

    # All LSTM intervention models (the summary point plots' "All Models"). The
    # time-course panels render EXP_ID only.
    MODELS = [
        "san-4601", "san-4602", "san-4603", "san-4604", "san-4605",
        "san-4606", "san-4615", "san-4616", "san-4617", "san-4618",
    ]

    # Type-hue palette for the summary point plots (DS4 flare over L2L/L2H/H2L/H2H).
    TYPE_ORDER = ["L2L", "L2H", "H2L", "H2H"]
    TYPE_PALETTE = get_color_palette(
        TYPE_ORDER, (("flare", 4),), linspace_range=np.array((0.0, 1.1))
    )

    FIGSIZE_TC = paper_style.PANEL_TUNING  # (2.5, 3.0) per-subplot; renderer widens
    FIGSIZE_PP = paper_style.PANEL_SQUARE  # (3.0, 3.0) point plots

    return (
        MODEL,
        EXP_ID,
        NUM_ALPHAS,
        ALPHAS,
        TIMESTEPS_PLOT,
        N_CONT,
        TIMESTEP_INTERVENTION,
        BOUNCE_IDX,
        FINAL_T_HZ,
        FINAL_T_CT,
        MODELS,
        TYPE_ORDER,
        TYPE_PALETTE,
        FIGSIZE_TC,
        FIGSIZE_PP,
        get_color_palette,
    )


@app.cell
def _(FuncFormatter, Line2D, math, np, paper_style, plt, sns, get_color_palette):
    # Render helpers (pure styling — SPEC rule 1: each panel self-contained).

    def render_interventions_rows(
        frame,
        exp_id,
        alphas,
        stat,
        title,
        vline,
        figsize,
        cond_order=("Low", "High"),
        hue="Alpha",
        x="Timestep",
        y="Value",
        ylabel="\nP(Color Change)",
        legend_width_inches=2.75,
    ):
        """Two condition-order subplots sharing one legend column (deck G/I/K/M).

        Ported from DS4 ``plot_interventions_rows`` (``:522``), rendering a
        single model. Only deviation from the source: ``stat_short`` comes from
        ``paper_style.SHORTENED_CONDITIONS`` (uppercase HZ/CT, SPEC ruling 6),
        not DS4's mixed-case ``'Hz'``/``'Cont'``. The apparent legend redundancy
        (a ref-cond legend built then overwritten by the 'Target' legend) is the
        source's behaviour and matches the deck: only Alpha / Target /
        Intervention render.
        """
        cond_order = list(cond_order)
        num_conds = len(cond_order)
        stat_short = paper_style.SHORTENED_CONDITIONS[stat]
        # dict_exp_alpha_mult: Low -> centroid 1, High -> centroid 0 (DS4).
        centroid_of = {"Low": 1, "High": 0}

        width_ratios = [figsize[0]] + [legend_width_inches] + [figsize[0]] * (num_conds - 1)
        total_width = sum(width_ratios)
        fig = plt.figure(figsize=(total_width, figsize[1]), constrained_layout=False)
        gs = fig.add_gridspec(1, num_conds + 1, width_ratios=width_ratios, wspace=0, hspace=0)
        axes = [fig.add_subplot(gs[0, 0])]
        for j in range(1, num_conds):
            axes.append(fig.add_subplot(gs[0, j + 1], sharey=axes[0], sharex=axes[0]))

        alphas_to_plot = alphas[alphas * -1 <= 0.0]
        try:
            palette = get_color_palette(
                sorted(alphas_to_plot, key=abs),
                (("viridis", len(alphas_to_plot)),),
                linspace_range=np.array((0.0, 1.1)),
            )
        except Exception:
            palette = "viridis"

        for j, stat_cond in enumerate(cond_order):
            ref_cond = "High" if stat_cond.lower() == "low" else "Low"
            ax = axes[j]
            centroid = centroid_of[stat_cond]
            sub = frame[
                (frame[stat] == stat_cond)
                & (frame["Centroid"] == centroid)
                & (frame[hue] * -1 <= 0.0)
            ]
            sns.lineplot(
                data=sub, x=x, y=y, hue=hue, palette=palette,
                errorbar="ci", seed=0,  # deterministic CI bands across re-runs
                ax=ax, legend=(j == num_conds - 1),
            )
            sns.lineplot(
                data=frame[(frame[stat] == ref_cond) & (frame[hue] == 0.0)],
                x=x, y=y, color="red", errorbar="ci", seed=0,
                ax=ax, legend=False, linewidth=2,
            )
            if vline:
                ax.axvline(x=vline, color="gray", linestyle="--", linewidth=2, alpha=0.7)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: int(v)))
            ax.set_title(f"{title}{stat_cond} to {ref_cond} {stat_short}")
            ax.set_ylabel(ylabel if j == 0 else "")

            if j == num_conds - 1:
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    labels, handles = zip(
                        *sorted(zip(labels, handles), key=lambda t: abs(float(t[0])))
                    )
                legend_title = f"{hue.title()}"
                bbox_x_anchor = -0.1 * (legend_width_inches / figsize[0]) - 0.1
                legend_alphas = ax.legend(
                    handles, labels, title=legend_title,
                    bbox_to_anchor=(bbox_x_anchor, 1.0275), loc="upper right",
                    frameon=True, ncol=2, columnspacing=1.0, handletextpad=0.5,
                )
                for text in legend_alphas.get_texts():
                    try:
                        text.set_text(f"{abs(float(text.get_text())):.1f}")
                    except ValueError:
                        pass
                ax.add_artist(legend_alphas)

                first_title_len = len(legend_title)
                num_rows = math.ceil(len(handles) / 2)
                # Ref-cond legend ("<ref> HZ") — built, then overwritten below
                # by the 'Target' legend (only the latter is add_artist'd). This
                # matches DS4 and the deck's Alpha/Target/Intervention column.
                offset = 0.05 + num_rows * 0.045 + 0.05
                ref_line = Line2D([0], [0], color="red", linewidth=2.5)
                ref_title = "Target"
                padding = max(0, first_title_len - len(ref_title))
                ref_legend_title = " " * (padding // 2) + ref_title + " " * (padding // 2)
                offset = 0.05 + len(handles) * 0.045 + 0.05
                legend_ref = ax.legend(
                    handles=[ref_line], labels=[""],
                    bbox_to_anchor=(bbox_x_anchor, 1.0 - offset), loc="upper right",
                    title=ref_legend_title, handlelength=2.5, frameon=True,
                )
                legend_ref.get_title().set_ha("center")
                ax.add_artist(legend_ref)

                if vline:
                    dashed_line = Line2D([0], [0], color="gray", linestyle="--", linewidth=2)
                    vline_title = "Intervention"
                    diff = first_title_len - len(vline_title)
                    pad_str = " " * (max(0, diff) // 2)
                    vline_title_legend = pad_str + vline_title + pad_str
                    offset_dashed = offset + 0.15
                    legend_dashed = ax.legend(
                        handles=[dashed_line], labels=[""],
                        bbox_to_anchor=(bbox_x_anchor, 1.0 - offset_dashed), loc="upper right",
                        title=vline_title_legend, handlelength=2.5, frameon=True,
                    )
                    legend_dashed.get_title().set_ha("center")
                    ax.add_artist(legend_dashed)

        fig.tight_layout()
        return fig

    def render_pointplot(frame, final_timestep, title, palette, figsize):
        """Summary point plot: P(Final Color Change) vs Alpha, hue=Type (H/J/L/N).

        Ported from DS4's polished FINAL point-plot cells (``:1176-1216`` etc.).
        Keeps seaborn's native ``hue='Type'`` legend (title 'Type') — DS4's N
        cell's stray ``plt.legend(loc='center left')`` that drops that title is
        NOT ported.
        """
        sub = frame[frame["Timestep"] == final_timestep]
        fig = plt.figure(figsize=figsize)
        ax = sns.pointplot(
            sub, x="Alpha", y="Value", hue="Type", palette=palette,
            seed=0,  # deterministic CI whiskers across re-runs
        )
        plt.xlabel("Alpha")
        plt.ylabel("P(Final Color Change)")

        n_ticks = len(ax.get_xticks())
        step = max(2, n_ticks // 5) if figsize[0] < 6 else (
            max(1, n_ticks // 8) if figsize[0] < 10 else 1
        )
        visible_ticks = range(0, n_ticks, step)
        ax.set_xticks(list(visible_ticks))
        unique_alphas = np.sort(sub["Alpha"].unique())
        tick_labels = [
            f"{unique_alphas[i]:.1f}" if i < len(unique_alphas) else "" for i in visible_ticks
        ]
        ax.set_xticklabels(tick_labels)
        plt.title(title)
        fig.tight_layout()
        return fig

    def add_type(frame, cond_col):
        # Type = <cond>2<centroid>, e.g. 'L2H' (DS4). cond_col is 'Hazard Rate'
        # or 'Contingency'; Centroid 0 -> 'L', else 'H'.
        frame = frame.copy()
        frame["Type"] = (
            frame[cond_col].apply(lambda x: "L" if x == "Low" else "H")
            + "2"
            + frame["Centroid"].apply(lambda x: "L" if x == 0 else "H")
        )
        return frame

    return render_interventions_rows, render_pointplot, add_type


# ---------------------------------------------------------------------------
# Hazard-rate block — straight trials (deck G/H, I/J)
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""## Hazard-rate block (straight trials)""")
    return


@app.cell
def _(
    ALPHAS,
    EXP_ID,
    MODEL,
    TIMESTEPS_PLOT,
    TIMESTEP_INTERVENTION,
    FIGSIZE_TC,
    paper_style,
    render_interventions_rows,
    transforms,
):
    def _():
        for unit, word, name in (
            ("hidden", "Hidden", "intervention_timecourse_hz_hidden"),  # G
            ("cell", "Cell", "intervention_timecourse_hz_cell"),        # I
        ):
            frame = transforms.intervention_prediction_frame(
                MODEL, EXP_ID, "hz", unit, num_alphas=len(ALPHAS), N=TIMESTEPS_PLOT
            )
            frame = frame[(frame["trial"] == "Straight") & (frame["idx_time"] == 2)]
            fig = render_interventions_rows(
                frame, EXP_ID, ALPHAS,
                stat="Hazard Rate",
                title=f"{word} Unit Intervention:\n",
                vline=TIMESTEP_INTERVENTION,
                figsize=FIGSIZE_TC,
            )
            paper_style.save_panel(fig, 6, name)

    _()
    return


@app.cell
def _(
    ALPHAS,
    FINAL_T_HZ,
    FIGSIZE_PP,
    MODEL,
    MODELS,
    TIMESTEPS_PLOT,
    TYPE_PALETTE,
    paper_style,
    pd,
    render_pointplot,
    transforms,
    add_type,
):
    def _():
        for unit, word, name in (
            ("hidden", "Hidden", "summary_pointplot_hz_hidden"),  # H
            ("cell", "Cell", "summary_pointplot_hz_cell"),        # J
        ):
            frames = []
            for exp_id in MODELS:
                f = transforms.intervention_prediction_frame(
                    MODEL, exp_id, "hz", unit, num_alphas=len(ALPHAS), N=TIMESTEPS_PLOT
                )
                f = f[(f["trial"] == "Straight") & (f["idx_time"] == 2)]
                frames.append(add_type(f, "Hazard Rate"))
            frame = pd.concat(frames, ignore_index=True)
            fig = render_pointplot(
                frame, FINAL_T_HZ,
                title=f"All Models {word}\nUnit Interventions",
                palette=TYPE_PALETTE, figsize=FIGSIZE_PP,
            )
            paper_style.save_panel(fig, 6, name)

    _()
    return


# ---------------------------------------------------------------------------
# Contingency block — wall-bounce trials (deck K/L, M/N)
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""## Contingency block (wall-bounce trials)""")
    return


@app.cell
def _(
    ALPHAS,
    BOUNCE_IDX,
    EXP_ID,
    MODEL,
    N_CONT,
    TIMESTEP_INTERVENTION,
    FIGSIZE_TC,
    paper_style,
    render_interventions_rows,
    transforms,
):
    def _():
        for unit, word, name in (
            ("hidden", "Hidden", "intervention_timecourse_ct_hidden"),  # K
            ("cell", "Cell", "intervention_timecourse_ct_cell"),        # M
        ):
            frame = transforms.intervention_prediction_frame(
                MODEL, EXP_ID, "cont", unit, num_alphas=len(ALPHAS), N=N_CONT
            )
            frame = frame[
                (frame["trial"] == "Bounce")
                & (frame["Timestep"] >= BOUNCE_IDX)
                & (frame["Contingency"].isin(["Low", "High"]))
            ].copy()
            frame["Timestep"] = frame["Timestep"] - BOUNCE_IDX
            fig = render_interventions_rows(
                frame, EXP_ID, ALPHAS,
                stat="Contingency",
                title=f"{word} Unit Intervention:\n",
                vline=TIMESTEP_INTERVENTION,
                figsize=FIGSIZE_TC,
            )
            paper_style.save_panel(fig, 6, name)

    _()
    return


@app.cell
def _(
    ALPHAS,
    BOUNCE_IDX,
    FINAL_T_CT,
    FIGSIZE_PP,
    MODEL,
    MODELS,
    N_CONT,
    TYPE_PALETTE,
    paper_style,
    pd,
    render_pointplot,
    transforms,
    add_type,
):
    def _():
        # Ruling 9: panel N's title says "Cell" (correcting the deck's published
        # "Hidden" copy-paste bug over what is genuinely the cell-unit
        # contingency frame). The word is taken straight from the unit here, so
        # both L (hidden) and N (cell) render the correct unit.
        for unit, word, name in (
            ("hidden", "Hidden", "summary_pointplot_ct_hidden"),  # L
            ("cell", "Cell", "summary_pointplot_ct_cell"),        # N (ruling 9)
        ):
            frames = []
            for exp_id in MODELS:
                f = transforms.intervention_prediction_frame(
                    MODEL, exp_id, "cont", unit, num_alphas=len(ALPHAS), N=N_CONT
                )
                f = f[
                    (f["trial"] == "Bounce")
                    & (f["Timestep"] >= BOUNCE_IDX)
                    & (f["Contingency"].isin(["Low", "High"]))
                ].copy()
                f["Timestep"] = f["Timestep"] - BOUNCE_IDX
                frames.append(add_type(f, "Contingency"))
            frame = pd.concat(frames, ignore_index=True)
            fig = render_pointplot(
                frame, FINAL_T_CT,
                title=f"All Models {word}\nUnit Interventions",
                palette=TYPE_PALETTE, figsize=FIGSIZE_PP,
            )
            paper_style.save_panel(fig, 6, name)

    _()
    return


if __name__ == "__main__":
    app.run()

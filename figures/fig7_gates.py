"""Figure 7 — network-gate panels (marimo, dual-use).

Ported into the paper-figure pipeline from three exploratory analysis notebooks
in the sibling ``hmdcpd-analysis`` repo (contract in
``tests/test_fig7_panels.py``). Five panels are rendered; panel A, the
hand-drawn "rescue" schematic, is composed externally:

    cell_unit_interventions_all_models.svg  (panel B, left)
        REUSED cell-unit intervention point plot — same rendered content as
        fig6's "All Models Cell Unit Interventions" panel, exported into fig7's
        own namespace because panel paths are per-figure and stable. Built via
        the shared, memoized ``transforms.intervention_prediction_frame``
        (landed with fig6). Its Type legend is too large to sit in either point
        plot, so it is stripped here and exported as its own panel below.

    interventions_legend.svg                (standalone Type legend)
        The Type (L2L/L2H/H2L/H2H) legend shared by both point plots, rendered
        alone on its own figure via ``paper_style.make_legend_panel`` so it can
        be placed independently when the figure is composed.

    gate_rescue_input_forget.svg            (panel B, right)
        Gate-frozen ("rescue") intervention point plot for the ``(i, f)`` gate
        pair — the only pair the paper shows. Built via the memoized
        ``transforms.gate_rescue_prediction_frame`` (gate-frozen "all-states"
        cache, N=29). The title formats the gate pair explicitly as
        ``(i, f)``; interpolating the raw Python tuple would render its repr
        with quotes.

    gate_scatter_delta_forget_input.svg           (bottom-left)
        Untitled single-exemplar-model (``san-4604``) delta-gate scatter,
        Delta Forget vs Delta Input, one point per (colour × unit).

    gate_scatter_delta_forget_input_unit_mean.svg (bottom-right)
        Aggregated unit-mean scatter, one point per (colour × model). No live
        source cell survives for this panel; it was reconstructed from an older
        copy of the source notebook, whose input frame is still live in the
        current one. Both scatters derive from one memoized
        ``transforms.gate_activity_delta_frame``: the single-model panel is the
        ``san-4604`` slice, the aggregated panel is the
        ``groupby(color_entered, model).mean()``.

Both point plots reuse fig6's render recipe, ported here rather than imported —
figure scripts stay independently readable and never import one another. Style
comes from ``paper_style.apply_style`` (live-text SVG); transform cells are
split from render cells; each panel of the composed figure is one
self-contained SVG via ``save_panel``.

Dual use: ``marimo edit figures/fig7_gates.py`` for interactive work; ``python
figures/fig7_gates.py`` runs every cell top-to-bottom and lands the 5 SVGs
under ``figures/panels/fig7/``.
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

    return np, paper_style, pd, plt, sns, transforms


@app.cell
def _(np, paper_style):
    # Style cell: applies the shared theme AND returns every render-time constant,
    # forcing style-before-render in marimo's DAG (svg.fonttype='none' set before
    # any panel is saved).
    paper_style.apply_style()

    # Shared colormap-palette helper (promoted to paper_style; one copy).
    get_color_palette = paper_style.get_color_palette

    MODEL = "lstm"

    # All LSTM intervention models — the "All Models" point plots (same set as
    # fig6). The single-model gate scatter uses san-4604, the exemplar the
    # source notebook hardcoded.
    MODELS = (
        "san-4601", "san-4602", "san-4603", "san-4604", "san-4605",
        "san-4606", "san-4615", "san-4616", "san-4617", "san-4618",
    )
    EXEMPLAR = "san-4604"

    NUM_ALPHAS = 11
    ALPHAS = np.linspace(0, 1, NUM_ALPHAS)

    N_HZ = 26          # cell-unit intervention window (same as fig6)
    N_RESCUE = 29      # gate-rescue window
    FINAL_T = 24       # final-timestep index the summary point plots collapse onto
    GATE_PAIR = ("i", "f")  # the only gate pair the paper shows

    # Type-hue palette for the summary point plots (flare over
    # L2L/L2H/H2L/H2H).
    TYPE_ORDER = ["L2L", "L2H", "H2L", "H2H"]
    TYPE_PALETTE = get_color_palette(
        TYPE_ORDER, (("flare", 4),), linspace_range=np.array((0.0, 1.1))
    )

    FIGSIZE_PP = paper_style.PANEL_SQUARE      # (3.0, 3.0) point plots
    FIGSIZE_SCATTER = paper_style.PANEL_SQUARE  # (3.0, 3.0) gate scatters
    return (
        ALPHAS,
        EXEMPLAR,
        FIGSIZE_PP,
        FIGSIZE_SCATTER,
        FINAL_T,
        GATE_PAIR,
        MODEL,
        MODELS,
        N_HZ,
        N_RESCUE,
        TYPE_PALETTE,
    )


@app.cell
def _(mo):
    # Export toggle. Every render cell displays its figures inline and writes the
    # SVGs only while this is on, so styling iterations in `marimo edit` need not
    # touch disk. It defaults on, which is what a headless
    # `python figures/fig7_gates.py` run sees — that run never touches the UI, so
    # it still lands all four panels.
    save_svgs = mo.ui.switch(value=True, label="Save SVG panels")
    save_svgs
    return (save_svgs,)


@app.cell
def _(np, plt, sns):
    # Render helpers (pure styling — each panel is self-contained: its own
    # axes, labels, and legend).

    def render_pointplot(frame, final_timestep, title, palette, figsize):
        """Summary point plot: P(Final Color Change) vs Alpha, hue=Type.

        Ported from the source notebooks' final point-plot cells (and fig6's
        copy), ported here rather than imported so each figure script stays
        independently readable. Draws full furniture by default and returns
        ``(fig, ax)`` — matching ``plot_cwc_swarm`` — so the render cell can
        choose per-panel decoration via ``paper_style.apply_decor``.
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
        return fig, ax

    def add_type(frame):
        # Type = <hazard>2<centroid>, e.g. 'L2H'. Centroid 0 -> 'L'.
        frame = frame.copy()
        frame["Type"] = (
            frame["Hazard Rate"].apply(lambda x: "L" if x == "Low" else "H")
            + "2"
            + frame["Centroid"].apply(lambda x: "L" if x == 0 else "H")
        )
        return frame

    def render_gate_scatter(frame, figsize, title=None, label_suffix=""):
        """Delta Forget (x='f') vs Delta Input (y='i') gate scatter.

        Ported verbatim from the source notebook's single-model cell; the
        reconstructed unit-mean variant adds the title and the ``(unit-mean)``
        label suffix. Legend: colour_entered hue (Blue/Green/Red) + the dashed
        Unity diagonal.
        """
        gate_x, gate_y = "f", "i"
        fig = plt.figure(figsize=figsize)
        ax = sns.scatterplot(
            data=frame, x=gate_x, y=gate_y, hue="color_entered",
            hue_order=["Blue", "Green", "Red"],
            palette={"Blue": "blue", "Green": "green", "Red": "red"},
            alpha=0.7,
        )
        max_abs = np.max([np.abs(ax.get_xlim()).max(), np.abs(ax.get_ylim()).max()])
        lims = [-max_abs, max_abs]
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1.5, alpha=0.75, zorder=0)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1.5, alpha=0.75, zorder=0)
        plt.plot(lims, lims, "k--", alpha=0.3, linewidth=1, zorder=0, label="Unity")
        # dict_gate_names: 'f' -> forget -> "Forget", 'i' -> input -> "Input".
        plt.xlabel(f"Delta Forget Gate Activity{label_suffix}")
        plt.ylabel(f"Delta Input Gate Activity{label_suffix}")
        plt.legend(loc="lower right")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        if title:
            plt.title(title)
        fig.tight_layout()
        return fig

    return add_type, render_gate_scatter, render_pointplot


@app.cell
def _(mo):
    mo.md(r"""
    ## Intervention point plots (panel B)
    """)
    return


@app.cell
def _(
    ALPHAS,
    FIGSIZE_PP,
    FINAL_T,
    MODEL,
    MODELS,
    N_HZ,
    TYPE_PALETTE,
    add_type,
    mo,
    paper_style,
    pd,
    render_pointplot,
    save_svgs,
    transforms,
):
    def _():
        # B, left — reused cell-unit interventions panel: same recipe as
        # fig6's summary_pointplot_hz_cell, exported under fig7's own name.
        frames = []
        for exp_id in MODELS:
            f = transforms.intervention_prediction_frame(
                MODEL, exp_id, "hz", "cell", num_alphas=len(ALPHAS), N=N_HZ
            )
            f = f[(f["trial"] == "Straight") & (f["idx_time"] == 2)]
            frames.append(add_type(f))
        frame = pd.concat(frames, ignore_index=True)
        fig, ax = render_pointplot(
            frame, FINAL_T,
            title="All Models Cell\nUnit Interventions",
            palette=TYPE_PALETTE, figsize=FIGSIZE_PP,
        )
        # The Type legend is too large to sit in either point plot, so it moves
        # to its own standalone panel: capture its handles BEFORE stripping it.
        handles, labels = ax.get_legend_handles_labels()
        paper_style.apply_decor(ax, paper_style.PanelDecor(legend=False))
        leg = paper_style.make_legend_panel(handles, labels, title="Type")
        if save_svgs.value:
            paper_style.save_panel(fig, 7, "cell_unit_interventions_all_models")
            paper_style.save_panel(leg, 7, "interventions_legend")
        return [fig, leg]

    _figs = _()
    mo.vstack(_figs)
    return


@app.cell
def _(
    ALPHAS,
    FIGSIZE_PP,
    FINAL_T,
    GATE_PAIR,
    MODEL,
    MODELS,
    N_RESCUE,
    TYPE_PALETTE,
    add_type,
    paper_style,
    pd,
    render_pointplot,
    save_svgs,
    transforms,
):
    def _():
        # B, right — (i, f) gate-frozen "rescue" interventions.
        frames = []
        for exp_id in MODELS:
            f = transforms.gate_rescue_prediction_frame(
                MODEL, exp_id, "hz", "cell", GATE_PAIR,
                num_alphas=len(ALPHAS), N=N_RESCUE,
            )
            f = f[(f["trial"] == "Straight") & (f["idx_time"] == 2)]
            frames.append(add_type(f))
        frame = pd.concat(frames, ignore_index=True)
        # Unquoted, comma-space gate-pair label, as published — interpolating
        # the raw tuple would render its repr, "('i', 'f')".
        pair = f"({GATE_PAIR[0]}, {GATE_PAIR[1]})"
        fig, ax = render_pointplot(
            frame, FINAL_T,
            title=f"All Models {pair} Gate\nRescue Interventions",
            palette=TYPE_PALETTE, figsize=FIGSIZE_PP,
        )
        # Legend lives in its own panel (see the left point plot); this panel
        # also drops its y-label — the two point plots need not share an
        # identical numeric range, so the left panel's y-label reads for both.
        paper_style.apply_decor(
            ax, paper_style.PanelDecor(legend=False, ylabel=None)
        )
        if save_svgs.value:
            paper_style.save_panel(fig, 7, "gate_rescue_input_forget")
        return fig

    _fig = _()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Delta gate-activity scatters (bottom row)
    """)
    return


@app.cell
def _(
    EXEMPLAR,
    FIGSIZE_SCATTER,
    MODEL,
    MODELS,
    mo,
    paper_style,
    render_gate_scatter,
    save_svgs,
    transforms,
):
    def _():
        # Both scatters derive from one memoized frame.
        plot_df = transforms.gate_activity_delta_frame(MODEL, MODELS)

        # bottom-left — single-model scatter, untitled.
        single = plot_df[plot_df["model"] == EXEMPLAR]
        fig_single = render_gate_scatter(single, figsize=FIGSIZE_SCATTER)
        if save_svgs.value:
            paper_style.save_panel(fig_single, 7, "gate_scatter_delta_forget_input")

        # bottom-right — aggregated unit-mean scatter: one point per
        # (colour_entered, model).
        agg = plot_df.groupby(["color_entered", "model"], as_index=False)[
            ["i", "f", "g", "o"]
        ].mean()
        fig_mean = render_gate_scatter(
            agg, figsize=FIGSIZE_SCATTER,
            title=(
                "Delta (High Hz - Low Hz) Forget vs Input"
                "\n Unit-Mean by Color Entered × Model"
            ),
            label_suffix=" (unit-mean)",
        )
        if save_svgs.value:
            paper_style.save_panel(
                fig_mean, 7, "gate_scatter_delta_forget_input_unit_mean"
            )

        return [fig_single, fig_mean]

    _figs = _()
    mo.vstack(_figs)
    return


if __name__ == "__main__":
    app.run()

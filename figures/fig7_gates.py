"""Figure 7 — network-gate panels (marimo, dual-use).

Ported into the paper-figure pipeline defined by ``figures/SPEC.md`` from three
``hmdcpd-analysis`` notebooks (page-1 scope of the fig7 deck
``google-drive/paper/png/Fig 7 - Network Gates-1.png``; contract in
``tests/test_fig7_panels.py``). Four panels are rendered (deck block A, the
hand-drawn "rescue" schematic, is Illustrator compose-time art — excluded):

    cell_unit_interventions_all_models.svg  (deck B, left)
        REUSED cell-unit intervention point plot — same rendered content as
        fig6's "All Models Cell Unit Interventions" panel, exported into fig7's
        own namespace (SPEC ruling 10 / rule 5). Ported from
        ``DS4-Interventions.py:1279-1313`` via the shared, memoized
        ``transforms.intervention_prediction_frame`` (landed with fig6).

    gate_rescue_input_forget.svg            (deck B, right)
        Gate-frozen ("rescue") intervention point plot for the ``(i, f)`` gate
        pair — the only pair in page-1 scope. Ported from
        ``DS6.2-Interventions-and-Gates.py:814-920`` via the new memoized
        ``transforms.gate_rescue_prediction_frame`` (gate-frozen "all-states"
        cache, N=29). The title uses the deck's unquoted ``(i, f)`` form rather
        than the source f-string's raw Python tuple repr (contract judgment call).

    gate_scatter_delta_forget_input.svg           (deck bottom-left)
        Untitled single-exemplar-model (``san-4604``) delta-gate scatter,
        Delta Forget vs Delta Input, one point per (colour × unit). Ported from
        ``DS6.4-Relative-Gate-Activities.py:568-590``.

    gate_scatter_delta_forget_input_unit_mean.svg (deck bottom-right)
        Aggregated unit-mean scatter, one point per (colour × model).
        RECONSTRUCTED per SPEC ruling 10 from the stale backup
        ``notebooks/marimo/DS6.4-Relative-Gate-Activities.py:1150-1195`` (no live
        DS6.4 cell survives); its input frame ``plot_df_1`` is still live at
        current ``DS6.4:678-696``. Both scatters derive from one memoized
        ``transforms.gate_activity_delta_frame`` (== ``plot_df_1``): the
        single-model panel is the ``san-4604`` slice, the aggregated panel is the
        ``groupby(color_entered, model).mean()``.

Both point plots reuse fig6's render recipe (ported here, not imported — SPEC
"port, never import"). Style comes from ``paper_style.apply_style`` (SPEC rule 2:
live-text SVG); transform cells are split from render cells; each deck box is
one self-contained SVG via ``save_panel`` (SPEC rule 1).

Dual use (SPEC rule 7): ``marimo edit figures/fig7_gates.py`` for interactive
work; ``python figures/fig7_gates.py`` runs every cell top-to-bottom and lands
the 4 SVGs under ``outputs/panels/fig7/``.
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
    # Style cell: applies the shared theme AND returns every render-time constant,
    # forcing style-before-render in marimo's DAG (svg.fonttype='none' set before
    # any panel is saved).
    paper_style.apply_style()

    def get_color_palette(columns, color_number_tup, linspace_range=(0.5, 1), linspace_offset=1):
        # Ported verbatim from the hmdcpd get_color_palette helper (as in fig6).
        import seaborn as _sns

        color_list = []
        for _i, (color, number) in enumerate(color_number_tup):
            cmap = _sns.color_palette(color, as_cmap=True)
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

    MODEL = "lstm"

    # All LSTM intervention models — the "All Models" point plots (same set as
    # fig6). The single-model gate scatter uses san-4604 (DS6.4's hardcoded
    # exemplar in df_ints_all).
    MODELS = (
        "san-4601", "san-4602", "san-4603", "san-4604", "san-4605",
        "san-4606", "san-4615", "san-4616", "san-4617", "san-4618",
    )
    EXEMPLAR = "san-4604"

    NUM_ALPHAS = 11
    ALPHAS = np.linspace(0, 1, NUM_ALPHAS)

    N_HZ = 26          # cell-unit intervention window (DS4/fig6)
    N_RESCUE = 29      # gate-rescue window (DS6.2)
    FINAL_T = 24       # final-timestep index the summary point plots collapse onto
    GATE_PAIR = ("i", "f")  # only gate pair in page-1 scope

    # Type-hue palette for the summary point plots (DS4/DS6.2 flare over
    # L2L/L2H/H2L/H2H).
    TYPE_ORDER = ["L2L", "L2H", "H2L", "H2H"]
    TYPE_PALETTE = get_color_palette(
        TYPE_ORDER, (("flare", 4),), linspace_range=np.array((0.0, 1.1))
    )

    FIGSIZE_PP = paper_style.PANEL_SQUARE      # (3.0, 3.0) point plots
    FIGSIZE_SCATTER = paper_style.PANEL_SQUARE  # (3.0, 3.0) gate scatters

    return (
        MODEL,
        MODELS,
        EXEMPLAR,
        ALPHAS,
        N_HZ,
        N_RESCUE,
        FINAL_T,
        GATE_PAIR,
        TYPE_PALETTE,
        FIGSIZE_PP,
        FIGSIZE_SCATTER,
    )


@app.cell
def _(np, plt, sns):
    # Render helpers (pure styling — SPEC rule 1: each panel self-contained).

    def render_pointplot(frame, final_timestep, title, palette, figsize):
        """Summary point plot: P(Final Color Change) vs Alpha, hue=Type.

        Ported from DS4's / DS6.2's polished FINAL point-plot cells (identical to
        fig6.render_pointplot — ported here, not imported, per SPEC).
        """
        sub = frame[frame["Timestep"] == final_timestep]
        fig = plt.figure(figsize=figsize)
        ax = sns.pointplot(sub, x="Alpha", y="Value", hue="Type", palette=palette)
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

    def add_type(frame):
        # Type = <hazard>2<centroid>, e.g. 'L2H' (DS4/DS6.2). Centroid 0 -> 'L'.
        frame = frame.copy()
        frame["Type"] = (
            frame["Hazard Rate"].apply(lambda x: "L" if x == "Low" else "H")
            + "2"
            + frame["Centroid"].apply(lambda x: "L" if x == 0 else "H")
        )
        return frame

    def render_gate_scatter(frame, figsize, title=None, label_suffix=""):
        """Delta Forget (x='f') vs Delta Input (y='i') gate scatter.

        Ported verbatim from DS6.4 (``:568-590`` single-model; the reconstructed
        ``:1170-1199`` backup adds the title and ``(unit-mean)`` label suffix).
        Legend: colour_entered hue (Blue/Green/Red) + the dashed Unity diagonal.
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

    return render_pointplot, add_type, render_gate_scatter


# ---------------------------------------------------------------------------
# Deck B — "All Models" intervention point plots
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""## Intervention point plots (deck B)""")
    return


@app.cell
def _(
    ALPHAS,
    FINAL_T,
    FIGSIZE_PP,
    MODEL,
    MODELS,
    N_HZ,
    TYPE_PALETTE,
    add_type,
    paper_style,
    pd,
    render_pointplot,
    transforms,
):
    def _():
        # B, left — reused cell-unit interventions panel (SPEC ruling 10; same
        # recipe as fig6's summary_pointplot_hz_cell, from DS4:1279-1313).
        frames = []
        for exp_id in MODELS:
            f = transforms.intervention_prediction_frame(
                MODEL, exp_id, "hz", "cell", num_alphas=len(ALPHAS), N=N_HZ
            )
            f = f[(f["trial"] == "Straight") & (f["idx_time"] == 2)]
            frames.append(add_type(f))
        frame = pd.concat(frames, ignore_index=True)
        fig = render_pointplot(
            frame, FINAL_T,
            title="All Models Cell\nUnit Interventions",
            palette=TYPE_PALETTE, figsize=FIGSIZE_PP,
        )
        paper_style.save_panel(fig, 7, "cell_unit_interventions_all_models")

    _()
    return


@app.cell
def _(
    ALPHAS,
    FINAL_T,
    FIGSIZE_PP,
    GATE_PAIR,
    MODEL,
    MODELS,
    N_RESCUE,
    TYPE_PALETTE,
    add_type,
    paper_style,
    pd,
    render_pointplot,
    transforms,
):
    def _():
        # B, right — (i, f) gate-frozen "rescue" interventions (DS6.2:814-920).
        frames = []
        for exp_id in MODELS:
            f = transforms.gate_rescue_prediction_frame(
                MODEL, exp_id, "hz", "cell", GATE_PAIR,
                num_alphas=len(ALPHAS), N=N_RESCUE,
            )
            f = f[(f["trial"] == "Straight") & (f["idx_time"] == 2)]
            frames.append(add_type(f))
        frame = pd.concat(frames, ignore_index=True)
        # Deck's unquoted, comma-space gate-pair label (NOT the source
        # f-string's raw tuple repr "('i', 'f')") — contract judgment call.
        pair = f"({GATE_PAIR[0]}, {GATE_PAIR[1]})"
        fig = render_pointplot(
            frame, FINAL_T,
            title=f"All Models {pair} Gate\nRescue Interventions",
            palette=TYPE_PALETTE, figsize=FIGSIZE_PP,
        )
        paper_style.save_panel(fig, 7, "gate_rescue_input_forget")

    _()
    return


# ---------------------------------------------------------------------------
# Deck bottom — delta gate-activity scatters
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""## Delta gate-activity scatters (deck bottom)""")
    return


@app.cell
def _(
    EXEMPLAR,
    FIGSIZE_SCATTER,
    MODEL,
    MODELS,
    paper_style,
    render_gate_scatter,
    transforms,
):
    def _():
        # Both scatters derive from one memoized frame (DS6.4 plot_df_1).
        plot_df = transforms.gate_activity_delta_frame(MODEL, MODELS)

        # bottom-left — single-model scatter, untitled (DS6.4:568-590).
        single = plot_df[plot_df["model"] == EXEMPLAR]
        fig = render_gate_scatter(single, figsize=FIGSIZE_SCATTER)
        paper_style.save_panel(fig, 7, "gate_scatter_delta_forget_input")

        # bottom-right — aggregated unit-mean scatter (reconstructed backup
        # DS6.4:1150-1195): one point per (colour_entered, model).
        agg = plot_df.groupby(["color_entered", "model"], as_index=False)[
            ["i", "f", "g", "o"]
        ].mean()
        fig = render_gate_scatter(
            agg, figsize=FIGSIZE_SCATTER,
            title=(
                "Delta (High Hz - Low Hz) Forget vs Input"
                "\n Unit-Mean by Color Entered × Model"
            ),
            label_suffix=" (unit-mean)",
        )
        paper_style.save_panel(fig, 7, "gate_scatter_delta_forget_input_unit_mean")

    _()
    return


if __name__ == "__main__":
    app.run()

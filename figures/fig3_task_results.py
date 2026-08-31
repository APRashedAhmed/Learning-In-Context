"""Figure 3 — task-results CWC panels (marimo, dual-use).

Six panels covering the paper's two trial types across three response sources
(participants, RNN, LSTM), exported individually under ``figures/panels/fig3/``:

* Straight-path trials — confidence-weighted choice against grayzone position,
  split by hazard rate on a light-to-dark blue ramp:
  ``cwc_straight_participants``, ``cwc_straight_rnn``, ``cwc_straight_lstm``.
  Every panel is self-contained, so each carries its own hazard-rate legend.
* Wall-bounce trials — confidence-weighted choice against contingency as a
  single red family; the x tick labels already name the levels, so these panels
  carry no legend: ``cwc_bounce_participants``, ``cwc_bounce_rnn``,
  ``cwc_bounce_lstm``.

Data sources:

* The model columns go through ``transforms.model_cwc_by_hazard`` /
  ``transforms.model_cwc_by_contingency``, one call per model type. Both sample
  a synthetic response pool sized to the surviving human cohort, so the pool
  size is read from the participant-count artifact rather than fixed, and the
  sampling seed is a single module constant shared by all four calls.
* The participants column goes through
  ``learning_in_context.analysis.participants.participant_cwc_by_hazard`` /
  ``participant_cwc_by_contingency``, fed the scored responses from
  ``data/cache/participants/participant_cwc.parquet`` and the full straight /
  bounce trial metadata. Grouping every participant against the same
  full-experiment buckets is what keeps a participant's rows independent of
  which other participants were analyzed alongside them.

This script renders; it never recomputes. The participant artifacts are read as
the participant_stats pipeline left them, and a missing artifact is an assertion
failure rather than a silent fallback.

Dual use: ``marimo edit figures/fig3_task_results.py`` for interactive work;
``python figures/fig3_task_results.py`` runs every cell top-to-bottom (via
``app.run()``) and lands all six SVGs.
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

    import json
    from pathlib import Path

    import pandas as pd

    from learning_in_context.analysis import participants
    from learning_in_context.visualization import cwc_plots, paper_style, transforms

    return Path, cwc_plots, json, paper_style, participants, pd, transforms


@app.cell
def _(Path, paper_style):
    # Style cell: applies the shared theme AND returns every render-time constant.
    # Render cells consume these, which forces style-before-render in marimo's
    # dependency DAG (so svg.fonttype='none' is set before any panel is saved).
    paper_style.apply_style()

    REPO_ROOT = Path(__file__).resolve().parents[1]
    PARTICIPANTS_DIR = REPO_ROOT / "data" / "cache" / "participants"

    # The model panels stand in for the human cohort on the very stimuli the
    # cohort saw, so they read that dataset and no other.
    DATASET = "participant_dataset"
    # One seed for every model CWC call, so the sampled pools cannot drift apart.
    CWC_SEED = 0

    HAZARD_ORDER = ("Low", "High")  # light-to-dark, and the legend's order
    CONTINGENCY_ORDER = ("Low", "Medium", "High")
    YLIM = (-1.05, 1.05)
    YLABEL = "CWC"
    FIGSIZE = paper_style.PANEL_SQUARE
    return (
        CONTINGENCY_ORDER,
        CWC_SEED,
        DATASET,
        FIGSIZE,
        HAZARD_ORDER,
        PARTICIPANTS_DIR,
        YLABEL,
        YLIM,
    )


@app.cell
def _(mo):
    # Export toggle. Every render cell displays its figure inline and writes the
    # SVG only while this is on, so styling iterations in `marimo edit` need not
    # touch disk. It defaults on, which is what a headless
    # `python figures/fig3_task_results.py` run sees — that run never touches
    # the UI, so it still lands all six panels.
    save_svgs = mo.ui.switch(value=True, label="Save SVG panels")
    mo.vstack([save_svgs])
    return (save_svgs,)


@app.cell
def _():
    # Render helpers (pure styling), shared by every panel that needs them.
    def style_gridlines(ax):
        """Dashed horizontal gridlines behind the swarm — this figure's reading aid."""
        ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
        ax.set_axisbelow(True)

    def anchor_legend(ax):
        """Pin the legend to the upper left corner of the axes.

        The renderer places legends automatically, which is data-driven and
        therefore lands in a different corner on each panel of a row; pinning it
        keeps the three straight-trial panels reading as one row.
        """
        title = ax.get_legend().get_title().get_text()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title=title, loc="upper left")

    return anchor_legend, style_gridlines


@app.cell
def _(PARTICIPANTS_DIR, json, participants):
    # The model pools are sized to the cohort that survived the participant
    # filters, so the size is read from the counts artifact — a literal would go
    # stale the next time the participant pipeline runs.
    _counts_path = PARTICIPANTS_DIR / participants.ARTIFACT_COUNTS
    assert _counts_path.exists(), (
        f"participant cohort counts not found at {_counts_path} — run the "
        "participant_stats pipeline task to produce the participant artifacts "
        "before rendering figure 3"
    )
    NUM_PARTICIPANTS = json.loads(_counts_path.read_text())["final_n"]
    return (NUM_PARTICIPANTS,)


@app.cell
def _(PARTICIPANTS_DIR, participants, pd):
    _cwc_path = PARTICIPANTS_DIR / participants.ARTIFACT_CWC
    assert _cwc_path.exists(), (
        f"scored participant responses not found at {_cwc_path} — run the "
        "participant_stats pipeline task to produce the participant artifacts "
        "before rendering figure 3"
    )
    # The artifact is one flat table; the CWC groupings take a mapping of
    # participant id to that participant's video-indexed responses.
    participant_cwc_dict = {
        _pid: _responses.set_index("Video ID")
        for _pid, _responses in pd.read_parquet(_cwc_path).groupby("Participant ID")
    }
    return (participant_cwc_dict,)


@app.cell
def _(DATASET, transforms):
    # The full trial metadata, split by trial type: every participant is grouped
    # against these same buckets, never against their own subset of trials.
    _df_meta = transforms.trial_metadata(DATASET)
    df_straight = _df_meta[_df_meta["trial"] == "Straight"]
    df_bounce = _df_meta[_df_meta["trial"] == "Bounce"]
    return df_bounce, df_straight


@app.cell
def _(mo):
    mo.md(r"""
    ## Straight Path Trials — CWC by grayzone position, split by hazard rate
    """)
    return


@app.cell
def _(df_straight, participant_cwc_dict, participants):
    df_participants_hazard = participants.participant_cwc_by_hazard(
        participant_cwc_dict, df_straight
    )
    return (df_participants_hazard,)


@app.cell
def _(CWC_SEED, DATASET, NUM_PARTICIPANTS, transforms):
    df_rnn_hazard = transforms.model_cwc_by_hazard(
        dataset=DATASET,
        model_types=("rnn",),
        num_participants=NUM_PARTICIPANTS,
        seed=CWC_SEED,
    )
    return (df_rnn_hazard,)


@app.cell
def _(CWC_SEED, DATASET, NUM_PARTICIPANTS, transforms):
    df_lstm_hazard = transforms.model_cwc_by_hazard(
        dataset=DATASET,
        model_types=("lstm",),
        num_participants=NUM_PARTICIPANTS,
        seed=CWC_SEED,
    )
    return (df_lstm_hazard,)


@app.cell
def _(
    FIGSIZE,
    HAZARD_ORDER,
    YLABEL,
    YLIM,
    anchor_legend,
    cwc_plots,
    df_participants_hazard,
    mo,
    paper_style,
    save_svgs,
    style_gridlines,
):
    _fig, _ax = cwc_plots.plot_cwc_swarm(
        df_participants_hazard,
        x="Grayzone Position",
        y="CWC",
        hue="Hazard Rate",
        figsize=FIGSIZE,
        ylim=YLIM,
        ylabel=YLABEL,
        hue_order=HAZARD_ORDER,
        palette_labels=HAZARD_ORDER,
        legend=True,
    )
    anchor_legend(_ax)
    style_gridlines(_ax)
    _fig.tight_layout()
    if save_svgs.value:
        paper_style.save_panel(_fig, 3, "cwc_straight_participants")
    mo.vstack([_fig])
    return


@app.cell
def _(
    FIGSIZE,
    HAZARD_ORDER,
    YLABEL,
    YLIM,
    anchor_legend,
    cwc_plots,
    df_rnn_hazard,
    mo,
    paper_style,
    save_svgs,
    style_gridlines,
):
    _fig, _ax = cwc_plots.plot_cwc_swarm(
        df_rnn_hazard,
        x="Grayzone Position",
        y="cwc",
        hue="Hazard Rate",
        figsize=FIGSIZE,
        ylim=YLIM,
        ylabel=YLABEL,
        hue_order=HAZARD_ORDER,
        palette_labels=HAZARD_ORDER,
        legend=True,
    )
    anchor_legend(_ax)
    style_gridlines(_ax)
    _fig.tight_layout()
    if save_svgs.value:
        paper_style.save_panel(_fig, 3, "cwc_straight_rnn")
    mo.vstack([_fig])
    return


@app.cell
def _(
    FIGSIZE,
    HAZARD_ORDER,
    YLABEL,
    YLIM,
    anchor_legend,
    cwc_plots,
    df_lstm_hazard,
    mo,
    paper_style,
    save_svgs,
    style_gridlines,
):
    _fig, _ax = cwc_plots.plot_cwc_swarm(
        df_lstm_hazard,
        x="Grayzone Position",
        y="cwc",
        hue="Hazard Rate",
        figsize=FIGSIZE,
        ylim=YLIM,
        ylabel=YLABEL,
        hue_order=HAZARD_ORDER,
        palette_labels=HAZARD_ORDER,
        legend=True,
    )
    anchor_legend(_ax)
    style_gridlines(_ax)
    _fig.tight_layout()
    if save_svgs.value:
        paper_style.save_panel(_fig, 3, "cwc_straight_lstm")
    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Wall Bounce Trials — CWC by contingency
    """)
    return


@app.cell
def _(df_bounce, participant_cwc_dict, participants):
    df_participants_bounce = participants.participant_cwc_by_contingency(
        participant_cwc_dict, df_bounce
    )
    return (df_participants_bounce,)


@app.cell
def _(CWC_SEED, DATASET, NUM_PARTICIPANTS, transforms):
    df_rnn_bounce = transforms.model_cwc_by_contingency(
        dataset=DATASET,
        model_types=("rnn",),
        num_participants=NUM_PARTICIPANTS,
        seed=CWC_SEED,
    )
    return (df_rnn_bounce,)


@app.cell
def _(CWC_SEED, DATASET, NUM_PARTICIPANTS, transforms):
    df_lstm_bounce = transforms.model_cwc_by_contingency(
        dataset=DATASET,
        model_types=("lstm",),
        num_participants=NUM_PARTICIPANTS,
        seed=CWC_SEED,
    )
    return (df_lstm_bounce,)


@app.cell
def _(
    CONTINGENCY_ORDER,
    FIGSIZE,
    YLABEL,
    YLIM,
    cwc_plots,
    df_participants_bounce,
    mo,
    paper_style,
    save_svgs,
    style_gridlines,
):
    _fig, _ax = cwc_plots.plot_cwc_swarm(
        df_participants_bounce,
        x="Contingency",
        y="CWC",
        figsize=FIGSIZE,
        ylim=YLIM,
        ylabel=YLABEL,
        hue_order=CONTINGENCY_ORDER,
    )
    style_gridlines(_ax)
    _fig.tight_layout()
    if save_svgs.value:
        paper_style.save_panel(_fig, 3, "cwc_bounce_participants")
    mo.vstack([_fig])
    return


@app.cell
def _(
    CONTINGENCY_ORDER,
    FIGSIZE,
    YLABEL,
    YLIM,
    cwc_plots,
    df_rnn_bounce,
    mo,
    paper_style,
    save_svgs,
    style_gridlines,
):
    _fig, _ax = cwc_plots.plot_cwc_swarm(
        df_rnn_bounce,
        x="Contingency",
        y="cwc",
        figsize=FIGSIZE,
        ylim=YLIM,
        ylabel=YLABEL,
        hue_order=CONTINGENCY_ORDER,
    )
    style_gridlines(_ax)
    _fig.tight_layout()
    if save_svgs.value:
        paper_style.save_panel(_fig, 3, "cwc_bounce_rnn")
    mo.vstack([_fig])
    return


@app.cell
def _(
    CONTINGENCY_ORDER,
    FIGSIZE,
    YLABEL,
    YLIM,
    cwc_plots,
    df_lstm_bounce,
    mo,
    paper_style,
    save_svgs,
    style_gridlines,
):
    _fig, _ax = cwc_plots.plot_cwc_swarm(
        df_lstm_bounce,
        x="Contingency",
        y="cwc",
        figsize=FIGSIZE,
        ylim=YLIM,
        ylabel=YLABEL,
        hue_order=CONTINGENCY_ORDER,
    )
    style_gridlines(_ax)
    _fig.tight_layout()
    if save_svgs.value:
        paper_style.save_panel(_fig, 3, "cwc_bounce_lstm")
    mo.vstack([_fig])
    return


if __name__ == "__main__":
    app.run()

"""Figure 2 — ideal-observer panels (marimo, dual-use).

Eight panels (contract in ``tests/test_fig2_panels.py``):

    Belief curves     estimate_curve_hazard_rate, estimate_curve_contingency
    Controlled rates  estimate_ctrl_hazard_change, estimate_ctrl_hazard_nochange,
                      estimate_ctrl_contingency_change,
                      estimate_ctrl_contingency_nochange
    Behaviour (CWC)   cwc_hazard_rate, cwc_contingency

The belief curves are the ideal Bayesian observer's per-frame probability that
the ball's colour has changed since it was last visible, drawn on a single
exemplar trial with one task parameter swept across the curves — the hazard
rate on a straight-path trial, the bounce contingency on a wall-bounce trial.

The controlled-rate panels are the *counting* observer's running Beta-posterior
estimate of a rate over a synthetic stimulus with events planted on exact
frames: the random hazard estimate over a stationary ball (with and without
colour changes), and the contingent estimate over a bouncing ball (with and
without colour changes coincident with the bounces).

The CWC panels are the Bayesian observer's end-of-trial responses, scored as
confidence-weighted choices and summarised per condition.

The two halves read different datasets, deliberately. The belief curves
describe the task's generative statistics, so they run on the control dataset
the belief-curve transform is anchored to, whose wall-bounce exemplar puts the
bounce in the middle of a fully occluded run — which is what makes the
contingency step legible. The CWC panels stand beside the human CWC panels of
figure 3, so they read the dataset the participants themselves saw and are
sampled to the size of the surviving cohort.

Dual use: ``marimo edit figures/fig2_ideal_observer.py`` for interactive work;
``python figures/fig2_ideal_observer.py`` runs every cell top-to-bottom (via
``app.run()``) and lands the eight SVGs under ``figures/panels/fig2/``.
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

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    from learning_in_context.analysis import participants
    from learning_in_context.visualization import cwc_plots, paper_style, transforms

    return (
        Path,
        cwc_plots,
        json,
        np,
        paper_style,
        participants,
        plt,
        sns,
        transforms,
    )


@app.cell
def _(paper_style):
    # Style cell: applies the shared theme AND returns every render-time constant.
    # Render cells consume these, which forces style-before-render in marimo's
    # dependency DAG (so svg.fonttype='none' is set before any panel is saved).
    paper_style.apply_style()

    # --- Estimate curves ---------------------------------------------------
    # Dataset the exemplar trials come from.
    CURVE_DATASET = "control_dataset"

    # Which of the matching trials supplies the geometry. Rank 0 changes colour
    # four frames before its occlusion begins, which spikes the curve inside the
    # visible lead-in and buries the occluded rise the panel is about; rank 1
    # has a quiet lead-in, so each curve starts flat at its swept parameter and
    # every later frame is the observer integrating across the occlusion.
    EXEMPLAR_RANK = 1

    # The hazard panel sweeps the transform's own default two levels; the
    # contingency panel sweeps the three bounce-contingency levels the task
    # manipulates.
    CONTINGENCY_LEVELS = (("Low", 0.1), ("Medium", 0.5), ("High", 0.9))

    # --- CWC panels --------------------------------------------------------
    # The dataset the human participants saw, so the observer's choices are
    # scored on exactly the trials figure 3's participant panels score.
    CWC_DATASET = "participant_dataset"

    # Base seed for the observer's response sampling, and for the jitter the
    # raw-point layer draws from numpy's global generator.
    SEED = 0

    # Per-sample CWC means sit within about a tenth of the axis of one another,
    # so a swarm cannot lay a category's samples out side by side: seaborn drops
    # every point it fails to place — over half of them in the busiest cell —
    # without failing. A threshold of zero takes the renderer's jittered-strip
    # path for every category, which draws every observation.
    CWC_SWARM_MAX = 0

    FIGSIZE = paper_style.PANEL_SQUARE

    # --- Controlled-input estimate panels ----------------------------------
    # The four controlled panels run the *counting* observer over a synthetic
    # stimulus with events planted on exact frames, so its running rate
    # estimates can be read against a known schedule. Retiming the whole panel
    # family is a one-line edit here: the length and the event frames are the
    # only geometry the panels bake in.
    CTRL_LENGTH = 90
    CTRL_EVENT_FRAMES = (30, 60)

    # Informative prior for the random-hazard estimate: Beta(19, 1) has mean 0.05,
    # a realistic baseline hazard, so the estimate starts near reality rather than
    # at the neutral 0.5. (Order is (no-change, change) pseudo-counts.)
    CTRL_HZ_PRIOR = (19, 1)
    return (
        CONTINGENCY_LEVELS,
        CTRL_EVENT_FRAMES,
        CTRL_HZ_PRIOR,
        CTRL_LENGTH,
        CURVE_DATASET,
        CWC_DATASET,
        CWC_SWARM_MAX,
        EXEMPLAR_RANK,
        FIGSIZE,
        SEED,
    )


@app.cell
def _(Path, json, participants):
    # Participants surviving the cohort exclusions. The observer is sampled to a
    # pool of this size, so a model swarm carries as many points as a human one.
    # Read from the counts artifact — the same source figure 3 sizes its model
    # pools from — so the two figures cannot drift apart when the participant
    # pipeline reruns.
    _counts_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "cache"
        / "participants"
        / participants.ARTIFACT_COUNTS
    )
    assert _counts_path.exists(), (
        f"participant cohort counts not found at {_counts_path} — run the "
        "participant_stats pipeline task to produce the participant artifacts "
        "before rendering figure 2"
    )
    NUM_PARTICIPANTS = json.loads(_counts_path.read_text())["final_n"]
    return (NUM_PARTICIPANTS,)


@app.cell
def _(mo):
    # Export toggle. Every render cell displays its figure inline and writes the
    # SVG only while this is on, so styling iterations in `marimo edit` need not
    # touch disk. It defaults on, which is what a headless
    # `python figures/fig2_ideal_observer.py` run sees — that run never touches
    # the UI, so it still lands all four panels.
    save_svgs = mo.ui.switch(value=True, label="Save SVG panels")
    save_svgs
    return (save_svgs,)


@app.cell
def _(paper_style, plt, sns):
    # Render helper (pure styling — each panel is self-contained: its own
    # axes, labels, and legend). The CWC panels need no helper of their own:
    # they are one call to the shared swarm renderer apiece.

    # The occluded stretch is drawn as a band behind the curves, echoing the
    # grey occluder the trials themselves show.
    OCCLUSION_SHADE = "0.9"

    def style_gridlines(ax):
        """Dashed horizontal gridlines behind the marks — the CWC family's reading aid."""
        ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="gray", alpha=0.5)
        ax.set_axisbelow(True)

    def anchor_legend(ax):
        """Pin the legend to the upper left corner of the axes.

        The renderer places legends automatically, which is data-driven and
        therefore drifts between panels; pinning it keeps this figure's CWC
        panel aligned with figure 3's row of the same family.
        """
        title = ax.get_legend().get_title().get_text()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title=title, loc="upper left")

    def render_belief_curves(
        df,
        family,
        linspace_range,
        legend_title,
        figsize,
        mark_endpoints=False,
        mark_bounce=False,
    ):
        levels = list(df["level"].cat.categories)
        palette = paper_style.get_color_palette(
            levels, ((family, len(levels)),), linspace_range=linspace_range
        )

        # Frame-indexed markers describe the trial, not a level, so they are
        # read off one level's rows.
        marks = df[df["level"] == levels[0]].sort_values("frame")
        frames = marks["frame"].to_numpy()
        occluded = marks["occluded"].to_numpy()

        fig, ax = plt.subplots(figsize=figsize)
        if occluded.any():
            ax.axvspan(
                frames[occluded].min(),
                frames[occluded].max(),
                color=OCCLUSION_SHADE,
                linewidth=0,
                zorder=0,
            )
        if mark_endpoints:
            # Where each grayzone position's occlusion ends — the frame at which
            # a trial of that position stops and asks for a response.
            for frame in marks["endpoint_offset"].dropna():
                ax.axvline(frame, linestyle="--", color="gray", linewidth=1, zorder=1)
        if mark_bounce:
            for frame in marks.loc[marks["is_bounce"], "frame"]:
                ax.axvline(frame, linestyle=":", color="black", linewidth=1, zorder=1)

        sns.lineplot(
            data=df,
            x="frame",
            y="p_change",
            hue="level",
            hue_order=levels,
            palette=palette,
            errorbar=None,  # one row per (level, frame): nothing to aggregate
            ax=ax,
            zorder=2,
        )
        ax.set_xlabel("Frame")
        ax.set_ylabel("P(Color Change)")
        ax.set_ylim(0, 1)
        sns.move_legend(ax, "upper left", title=legend_title, frameon=True)
        fig.tight_layout()
        return fig

    def render_estimate_curve(df, color, ylabel, figsize, ylim=None):
        """One counting-observer rate-estimate curve over a controlled input.

        Unlike the belief curves, these are not on a fixed 0–1 axis: the random
        (hazard) estimate collapses towards the low empirical rate, so its steps
        would vanish on a 0–1 axis. The y-axis is left autoscaled and only its
        floor is pinned to zero, which keeps a rate reading as a probability
        while letting each panel show its own dynamic range. Event frames — the
        colour changes or bounces that drive this channel — are marked with a
        dotted vertical line; the estimate steps one frame later, when the
        observer banks the event.
        """
        marks = df.sort_values("frame")
        fig, ax = plt.subplots(figsize=figsize)
        for frame in marks.loc[marks["is_event"], "frame"]:
            ax.axvline(frame, linestyle=":", color="black", linewidth=1, zorder=1)
        sns.lineplot(
            data=marks,
            x="frame",
            y="estimate",
            color=color,
            errorbar=None,  # one row per frame: nothing to aggregate
            ax=ax,
            zorder=2,
        )
        ax.set_xlabel("Frame")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
        fig.tight_layout()
        return fig

    return (
        anchor_legend,
        render_belief_curves,
        render_estimate_curve,
        style_gridlines,
    )


@app.cell
def _ctrl_hz_ylim(CTRL_EVENT_FRAMES, CTRL_HZ_PRIOR, CTRL_LENGTH, transforms):
    # Shared y-limits for the two hazard panels so they read on one axis. Both
    # curves start at the prior mean; the limit follows the larger with headroom.
    _hz_change_curve = transforms.ideal_observer_estimate_curves(
        channel="hz",
        length=CTRL_LENGTH,
        color_change_frames=CTRL_EVENT_FRAMES,
        prior_pccnvc=CTRL_HZ_PRIOR,
    )
    _hz_nochange_curve = transforms.ideal_observer_estimate_curves(
        channel="hz",
        length=CTRL_LENGTH,
        prior_pccnvc=CTRL_HZ_PRIOR,
    )
    CTRL_HZ_YLIM = (
        0.0,
        1.1 * max(_hz_change_curve["estimate"].max(), _hz_nochange_curve["estimate"].max()),
    )
    return (CTRL_HZ_YLIM,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Estimate curves — the observer's belief across one trial
    """)
    return


@app.cell
def _(CURVE_DATASET, EXEMPLAR_RANK, transforms):
    # Transform (memoized): hazard sweep over a straight-path exemplar, at the
    # transform's own default two hazard levels.
    df_curve_hz = transforms.ideal_observer_belief_curves(
        dataset=CURVE_DATASET,
        trial_type="Straight",
        sweep="hazard",
        exemplar_rank=EXEMPLAR_RANK,
    )
    return (df_curve_hz,)


@app.cell
def _(CONTINGENCY_LEVELS, CURVE_DATASET, EXEMPLAR_RANK, transforms):
    # Transform (memoized): contingency sweep over a wall-bounce exemplar —
    # only bounce trials carry the event the contingency governs.
    df_curve_ct = transforms.ideal_observer_belief_curves(
        dataset=CURVE_DATASET,
        trial_type="Bounce",
        sweep="contingency",
        levels=CONTINGENCY_LEVELS,
        exemplar_rank=EXEMPLAR_RANK,
    )
    return (df_curve_ct,)


@app.cell
def _(
    FIGSIZE,
    cwc_plots,
    df_curve_hz,
    mo,
    paper_style,
    render_belief_curves,
    save_svgs,
):
    def _():
        # One colour grammar across the figure: the hazard curves take the blue
        # split ramp the CWC renderer uses for its own two-level split.
        fig = render_belief_curves(
            df_curve_hz,
            family=cwc_plots.SPLIT_FAMILY,
            linspace_range=cwc_plots.SPLIT_LINSPACE,
            legend_title="Hazard Rate",
            figsize=FIGSIZE,
            mark_endpoints=True,
        )
        if save_svgs.value:
            paper_style.save_panel(fig, 2, "estimate_curve_hazard_rate")
        return fig

    _fig = _()
    mo.vstack([_fig])
    return


@app.cell
def _(
    FIGSIZE,
    cwc_plots,
    df_curve_ct,
    mo,
    paper_style,
    render_belief_curves,
    save_svgs,
):
    def _():
        # The contingency curves take the CWC renderer's red family, sampled
        # over the split range so three levels read light-to-dark.
        fig = render_belief_curves(
            df_curve_ct,
            family=cwc_plots.FAMILY,
            linspace_range=cwc_plots.SPLIT_LINSPACE,
            legend_title="Contingency",
            figsize=FIGSIZE,
            mark_bounce=True,
        )
        if save_svgs.value:
            paper_style.save_panel(fig, 2, "estimate_curve_contingency")
        return fig

    _fig = _()
    mo.vstack([_fig])
    return


@app.cell
def _ctrl_header(mo):
    mo.md(r"""
    ## Controlled-input estimate panels — the counting observer's running rates

    The counting observer accumulates evidence over a synthetic stimulus with
    events planted on exact frames, and reports its Beta-posterior estimate of a
    rate at every frame. The random (hazard, `pccnvc`) estimate collapses towards
    the low empirical rate with a step up at each colour change; the contingent
    (`pccovc`) estimate steps up at a bounce that changes colour and down at one
    that does not. Each estimate steps one frame after its event, when the observer
    banks it.
    """)
    return


@app.cell
def _(
    CTRL_EVENT_FRAMES,
    CTRL_HZ_PRIOR,
    CTRL_HZ_YLIM,
    CTRL_LENGTH,
    FIGSIZE,
    cwc_plots,
    mo,
    paper_style,
    render_estimate_curve,
    save_svgs,
    transforms,
):
    def _():
        # Hazard estimate over a stationary ball whose colour changes at the
        # event frames: with no velocity event, every change routes to the
        # random channel, so the estimate steps up one frame after each. The
        # informative prior starts the estimate near a realistic hazard rate.
        df = transforms.ideal_observer_estimate_curves(
            channel="hz",
            length=CTRL_LENGTH,
            color_change_frames=CTRL_EVENT_FRAMES,
            prior_pccnvc=CTRL_HZ_PRIOR,
        )
        color = paper_style.get_color_palette(
            ["hz"], ((cwc_plots.SPLIT_FAMILY, (2, 1)),), linspace_range=cwc_plots.SPLIT_LINSPACE
        )["hz"]
        fig = render_estimate_curve(
            df, color, "Estimated Hazard Rate", FIGSIZE, ylim=CTRL_HZ_YLIM
        )
        if save_svgs.value:
            paper_style.save_panel(fig, 2, "estimate_ctrl_hazard_change")
        return fig

    _fig = _()
    mo.vstack([_fig])
    return


@app.cell
def _(
    CTRL_HZ_PRIOR,
    CTRL_HZ_YLIM,
    CTRL_LENGTH,
    FIGSIZE,
    cwc_plots,
    mo,
    paper_style,
    render_estimate_curve,
    save_svgs,
    transforms,
):
    def _():
        # Hazard estimate over a stationary ball with no colour changes at all:
        # the observer banks only failures, so the estimate decays from the prior
        # mean towards zero. Shares the change panel's axis for direct comparison.
        df = transforms.ideal_observer_estimate_curves(
            channel="hz",
            length=CTRL_LENGTH,
            prior_pccnvc=CTRL_HZ_PRIOR,
        )
        color = paper_style.get_color_palette(
            ["hz"], ((cwc_plots.SPLIT_FAMILY, (2, 1)),), linspace_range=cwc_plots.SPLIT_LINSPACE
        )["hz"]
        fig = render_estimate_curve(
            df, color, "Estimated Hazard Rate", FIGSIZE, ylim=CTRL_HZ_YLIM
        )
        if save_svgs.value:
            paper_style.save_panel(fig, 2, "estimate_ctrl_hazard_nochange")
        return fig

    _fig = _()
    mo.vstack([_fig])
    return


@app.cell
def _(
    CTRL_EVENT_FRAMES,
    CTRL_LENGTH,
    FIGSIZE,
    cwc_plots,
    mo,
    paper_style,
    render_estimate_curve,
    save_svgs,
    transforms,
):
    def _():
        # Contingent estimate over a bouncing ball whose colour changes at each
        # bounce: every bounce is a success, so the estimate steps up one frame
        # after each (0.5 → 2/3 → 3/4 under the neutral prior). Full 0–1 axis.
        df = transforms.ideal_observer_estimate_curves(
            channel="cont",
            length=CTRL_LENGTH,
            bounce_frames=CTRL_EVENT_FRAMES,
            color_change_frames=CTRL_EVENT_FRAMES,
        )
        color = paper_style.get_color_palette(
            ["cont"], ((cwc_plots.FAMILY, (3, 2)),), linspace_range=cwc_plots.SPLIT_LINSPACE
        )["cont"]
        fig = render_estimate_curve(df, color, "Estimated Contingency", FIGSIZE, ylim=(0, 1))
        if save_svgs.value:
            paper_style.save_panel(fig, 2, "estimate_ctrl_contingency_change")
        return fig

    _fig = _()
    mo.vstack([_fig])
    return


@app.cell
def _(
    CTRL_EVENT_FRAMES,
    CTRL_LENGTH,
    FIGSIZE,
    cwc_plots,
    mo,
    paper_style,
    render_estimate_curve,
    save_svgs,
    transforms,
):
    def _():
        # Contingent estimate over a bouncing ball whose colour never changes:
        # every bounce is a failure, so the estimate steps down one frame after
        # each (0.5 → 1/3 → 1/4 under the neutral prior). Full 0–1 axis.
        df = transforms.ideal_observer_estimate_curves(
            channel="cont",
            length=CTRL_LENGTH,
            bounce_frames=CTRL_EVENT_FRAMES,
        )
        color = paper_style.get_color_palette(
            ["cont"], ((cwc_plots.FAMILY, (3, 2)),), linspace_range=cwc_plots.SPLIT_LINSPACE
        )["cont"]
        fig = render_estimate_curve(df, color, "Estimated Contingency", FIGSIZE, ylim=(0, 1))
        if save_svgs.value:
            paper_style.save_panel(fig, 2, "estimate_ctrl_contingency_nochange")
        return fig

    _fig = _()
    mo.vstack([_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Confidence-weighted choice — the observer's responses
    """)
    return


@app.cell
def _(CWC_DATASET, NUM_PARTICIPANTS, SEED, transforms):
    # Transform (memoized): mean CWC per hazard rate and grayzone position over
    # straight-path trials, for the ideal observer alone.
    df_cwc_hz = transforms.model_cwc_by_hazard(
        dataset=CWC_DATASET,
        model_types=("ibo",),
        num_participants=NUM_PARTICIPANTS,
        seed=SEED,
    )
    return (df_cwc_hz,)


@app.cell
def _(CWC_DATASET, NUM_PARTICIPANTS, SEED, transforms):
    # Transform (memoized): mean CWC per contingency over wall-bounce trials.
    df_cwc_ct = transforms.model_cwc_by_contingency(
        dataset=CWC_DATASET,
        model_types=("ibo",),
        num_participants=NUM_PARTICIPANTS,
        seed=SEED,
    )
    return (df_cwc_ct,)


@app.cell
def _(
    CWC_SWARM_MAX,
    FIGSIZE,
    SEED,
    anchor_legend,
    cwc_plots,
    df_cwc_hz,
    mo,
    np,
    paper_style,
    save_svgs,
    style_gridlines,
):
    def _():
        # seaborn draws the jittered raw-point layer from numpy's global
        # generator, so it is pinned per panel: re-exporting an unchanged panel
        # must be byte-identical.
        np.random.seed(SEED)
        fig, _ax = cwc_plots.plot_cwc_swarm(
            df_cwc_hz,
            x="Grayzone Position",
            y="cwc",
            hue="Hazard Rate",
            hue_order=["Low", "High"],  # light-to-dark, weakest level first
            figsize=FIGSIZE,
            ylabel="CWC",
            swarm_max=CWC_SWARM_MAX,
        )
        anchor_legend(_ax)
        style_gridlines(_ax)
        fig.tight_layout()
        if save_svgs.value:
            paper_style.save_panel(fig, 2, "cwc_hazard_rate")
        return fig

    _fig = _()
    mo.vstack([_fig])
    return


@app.cell
def _(
    CWC_SWARM_MAX,
    FIGSIZE,
    SEED,
    cwc_plots,
    df_cwc_ct,
    mo,
    np,
    paper_style,
    save_svgs,
    style_gridlines,
):
    def _():
        # No `hue`: the x-categories are the contingency levels themselves, so
        # the panel is drawn in the renderer's family mode — one colour, one
        # mean trace, and no legend, because the tick labels name the levels.
        np.random.seed(SEED)
        fig, _ax = cwc_plots.plot_cwc_swarm(
            df_cwc_ct,
            x="Contingency",
            y="cwc",
            hue_order=["Low", "Medium", "High"],
            figsize=FIGSIZE,
            ylabel="CWC",
            swarm_max=CWC_SWARM_MAX,
        )
        style_gridlines(_ax)
        fig.tight_layout()
        if save_svgs.value:
            paper_style.save_panel(fig, 2, "cwc_contingency")
        return fig

    _fig = _()
    mo.vstack([_fig])
    return


if __name__ == "__main__":
    app.run()

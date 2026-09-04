"""Figure 4 — identifying critical units (marimo, dual-use).

Ports the ElasticNet critical-units panels from the exploratory analysis
notebooks in the sibling ``hmdcpd-analysis`` repo. Four panels — B/C/E/F of
figure 4; A/D/G are hand-drawn schematics composed externally, so no script
produces them:

    B  score_curves_hazard_rate.svg   — F1/Accuracy vs. ElasticNet Alpha (binary)
    C  coef_heatmap_hazard_rate.svg   — 32-unit coefficient heatmap
    E  score_curves_contingency.svg   — per-label F1/Accuracy (3-class decode)
    F  coef_heatmap_contingency.svg   — 32-unit coefficient heatmap

Dual use: ``marimo edit figures/fig4_identifying_units.py`` for interactive
work; ``python figures/fig4_identifying_units.py`` runs every cell top-to-bottom
(via ``app.run()``) and lands the 4 SVGs under ``figures/panels/fig4/``.

Two deliberate corrections to the ported source
-----------------------------------------------
1. **Dead-renderer fix.** The source's ``plot_coefs_and_metrics`` references
   unbound ``_coefs``/``_hline_chance`` locals (a 2-line typo) and would
   ``NameError`` at call time; a later copy of the same function in the sibling
   notebooks binds the parameters correctly. The render helpers below port that
   fixed form, split into two self-contained panels.
2. **Metrics correction.** The source's contingency render cell reuses
   ``metrics_to_plot=['accuracy']`` left over from the hazard cell, which alone
   would not draw the per-label ``F1 - Label 0/1/2`` lines even though the
   ``average=None`` f1 data exists. Both panels here pass F1 *and* Accuracy;
   the per-stat f1 averaging (binary scalar for hz, per-label vector for the
   3-class contingency decode) is what makes hz render a single ``F1`` line and
   contingency render three.

Scope note: only the exemplar model (``san-4604``) is fitted — the ~100
ElasticNet fits behind these four panels — not the source's full 10-model loop.
These panels show a single model; the multi-model aggregation is a hand-drawn
schematic. These fits are the pipeline's one tolerated piece of inline compute,
memoized via ``transforms.elasticnet_coefficient_paths`` so warm reruns are
instant.
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
    import warnings

    import matplotlib

    matplotlib.use("Agg")  # headless: never reach for a GUI backend

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # The elastic-net logistic sweep does not converge at every C; that is
    # expected on this recipe (the source notebooks ran it the same way).
    # Silence the flood so a headless run's stderr stays readable — it does not
    # affect the fit paths.
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*alpha=0.*")

    from learning_in_context.visualization import paper_style
    from learning_in_context.visualization import transforms

    return TwoSlopeNorm, inset_axes, np, paper_style, plt, transforms


@app.cell
def _(paper_style):
    # Style cell: applies the shared theme AND returns render-time constants.
    # Render cells consume these, forcing style-before-render in marimo's DAG
    # (so svg.fonttype='none' is set before any panel is saved).
    paper_style.apply_style()

    # Pinned to the exemplar model (san-4604, lstm, extended_dataset) — the
    # single model whose panels B/C/E/F appear in figure 4.
    DATASET = "extended_dataset"
    MODEL = "lstm"
    EXP_ID = "san-4604"

    # Panel sizes from the paper_style size vocabulary, overridden per panel:
    # score curves are wide-and-short, heatmaps taller. Panels are exported at
    # final physical size, so these inches are what the composed figure gets.
    FIGSIZE_SCORE = (paper_style.HALF_WIDTH, 1.7)
    FIGSIZE_HEATMAP = (paper_style.HALF_WIDTH, 2.6)
    return DATASET, EXP_ID, FIGSIZE_HEATMAP, FIGSIZE_SCORE, MODEL


@app.cell
def _(mo):
    # Export toggle. Every render cell displays its figure inline and writes the
    # SVG only while this is on, so styling iterations in `marimo edit` need not
    # touch disk. It defaults on, which is what a headless
    # `python figures/fig4_identifying_units.py` run sees — that run never
    # touches the UI, so it still lands all four panels.
    save_svgs = mo.ui.switch(value=True, label="Save SVG panels")
    save_svgs
    return (save_svgs,)


@app.cell
def _(np, plt):
    # Render helper — score curve (ported from the source's
    # plot_coefs_and_metrics ax1 block, in its fixed form; one self-contained
    # panel with its own axes, labels, and legend).
    def render_score_curve(
        metrics,
        C_logspace,
        metrics_to_plot,
        hline_chance,
        vline_performance,
        figsize,
        xlabel="ElasticNet Alpha",
    ):
        metric_arrays = {
            name.title(): np.asarray(metrics[name]) for name in metrics_to_plot
        }

        fig, ax = plt.subplots(figsize=figsize)
        for label, values in metric_arrays.items():
            if values.ndim > 1:
                for i, sub in enumerate(values.T):
                    ax.plot(C_logspace, sub, label=f"{label} - Label {i}")
            else:
                ax.plot(C_logspace, values, label=label)

        if hline_chance:
            ax.axhline(
                hline_chance,
                color="grey",
                ls="--",
                label=f"Chance: {int(hline_chance * 100)}%",
            )
        if vline_performance is not None:
            idx, metric_name = vline_performance
            metric_value = metric_arrays[metric_name.title()][idx]
            ax.axvline(
                C_logspace[idx],
                color="grey",
                ls="-.",
                label=f"{metric_name.title()}: {int(metric_value * 100)}%",
            )

        ax.set_xscale("log")
        ax.set_xlim(C_logspace[0], C_logspace[-1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Score")
        ax.legend(loc="lower left", fontsize="xx-small")
        fig.tight_layout()
        return fig

    return (render_score_curve,)


@app.cell
def _(TwoSlopeNorm, inset_axes, np, plt):
    # Render helper — coefficient heatmap (ported from the source's ax3 block,
    # in its fixed form; colorbar labelled "Coefficient Value" to match the
    # composed figure and this repo's critical_units_plots.py, rather than the
    # source's blank label).
    def render_coef_heatmap(
        coefs,
        C_logspace,
        figsize,
        cmap="seismic",
        ylabel_show_every=2,
        vline_index=None,
        xlabel="ElasticNet Alpha",
        heatmap_ylabel="(H)idden / (C)ell Unit Number",
    ):
        coefs = np.asarray(coefs)  # (n_units, n_alphas)
        N, _L = coefs.shape
        unit_number = [
            f"{state}{number:0>2}"
            for state in ["H", "C"]
            for number in range(N // 2)
        ]

        abs_max = max(abs(coefs.min()), abs(coefs.max())) or 1.0
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

        fig, ax = plt.subplots(figsize=figsize)
        X, Y = np.meshgrid(C_logspace, np.arange(N))
        im = ax.pcolor(X, Y, coefs, cmap=cmap, norm=norm)

        ax.set_yticks(np.arange(0, N, ylabel_show_every))
        ax.set_yticklabels([unit_number[i] for i in range(0, N, ylabel_show_every)])
        ax.set_ylabel(heatmap_ylabel)
        ax.set_xscale("log")
        ax.set_xlim(C_logspace[0], C_logspace[-1])
        ax.set_xlabel(xlabel)

        if vline_index is not None:
            ax.axvline(C_logspace[vline_index], color="grey", ls="-.")

        axins = inset_axes(ax, height="97%", width="2.5%", loc="right")
        cb = fig.colorbar(im, cax=axins, orientation="vertical")
        axins.yaxis.set_ticks_position("left")
        cb.set_label("Coefficient Value", labelpad=10)

        fig.tight_layout()
        return fig

    return (render_coef_heatmap,)


@app.cell
def _(DATASET, EXP_ID, MODEL, transforms):
    # Hazard-rate: binary elastic-net logistic decode (single F1 line).
    fit_hz = transforms.elasticnet_coefficient_paths(
        dataset=DATASET,
        model_name=MODEL,
        exp_id=EXP_ID,
        stat="hz",
    )
    return (fit_hz,)


@app.cell
def _(DATASET, EXP_ID, MODEL, transforms):
    # Contingency: 3-class decode cast as elastic-net regression (per-label F1).
    fit_cont = transforms.elasticnet_coefficient_paths(
        dataset=DATASET,
        model_name=MODEL,
        exp_id=EXP_ID,
        stat="cont_r",
    )
    return (fit_cont,)


@app.cell
def _(fit_cont, fit_hz, np):
    # Vline index = last non-chance alpha (offset -3 for hz, -2 for cont, as
    # in the source notebooks).
    def _last_non_chance(metrics, offset):
        acc = np.asarray(metrics["accuracy"])
        return int(np.argmin(np.round(acc, 3))) - offset

    VLINE_HZ = _last_non_chance(fit_hz["metrics"], 3)
    VLINE_CONT = _last_non_chance(fit_cont["metrics"], 2)
    return VLINE_CONT, VLINE_HZ


@app.cell
def _(mo):
    mo.md(r"""
    ## Panels B/C — Hazard rate
    """)
    return


@app.cell
def _(
    FIGSIZE_SCORE,
    VLINE_HZ,
    fit_hz,
    paper_style,
    render_score_curve,
    save_svgs,
):
    def _():
        fig = render_score_curve(
            metrics=fit_hz["metrics"],
            C_logspace=fit_hz["C_logspace"],
            metrics_to_plot=["f1", "accuracy"],  # F1 first → legend order
            hline_chance=0.5,
            vline_performance=(VLINE_HZ, "accuracy"),
            figsize=FIGSIZE_SCORE,
        )
        if save_svgs.value:
            paper_style.save_panel(fig, 4, "score_curves_hazard_rate")
        return fig

    _fig = _()
    _fig
    return


@app.cell
def _(
    FIGSIZE_HEATMAP,
    VLINE_HZ,
    fit_hz,
    paper_style,
    render_coef_heatmap,
    save_svgs,
):
    def _():
        fig = render_coef_heatmap(
            coefs=fit_hz["coefs"],
            C_logspace=fit_hz["C_logspace"],
            figsize=FIGSIZE_HEATMAP,
            vline_index=VLINE_HZ,
        )
        if save_svgs.value:
            paper_style.save_panel(fig, 4, "coef_heatmap_hazard_rate")
        return fig

    _fig = _()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Panels E/F — Contingency
    """)
    return


@app.cell
def _(
    FIGSIZE_SCORE,
    VLINE_CONT,
    fit_cont,
    paper_style,
    render_score_curve,
    save_svgs,
):
    def _():
        fig = render_score_curve(
            metrics=fit_cont["metrics"],
            C_logspace=fit_cont["C_logspace"],
            metrics_to_plot=["f1", "accuracy"],  # f1 is (n_alphas, 3) → 3 lines
            hline_chance=1 / 3,
            vline_performance=(VLINE_CONT, "accuracy"),
            figsize=FIGSIZE_SCORE,
        )
        if save_svgs.value:
            paper_style.save_panel(fig, 4, "score_curves_contingency")
        return fig

    _fig = _()
    _fig
    return


@app.cell
def _(
    FIGSIZE_HEATMAP,
    VLINE_CONT,
    fit_cont,
    paper_style,
    render_coef_heatmap,
    save_svgs,
):
    def _():
        fig = render_coef_heatmap(
            coefs=fit_cont["coefs"],
            C_logspace=fit_cont["C_logspace"],
            figsize=FIGSIZE_HEATMAP,
            vline_index=VLINE_CONT,
        )
        if save_svgs.value:
            paper_style.save_panel(fig, 4, "coef_heatmap_contingency")
        return fig

    _fig = _()
    _fig
    return


if __name__ == "__main__":
    app.run()

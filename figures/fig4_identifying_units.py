"""Figure 4 — identifying critical units (marimo, dual-use).

Ports the ElasticNet critical-units panels from
``hmdcpd-analysis/notebooks/DS2-Identifying-Critical-Units.py`` into the
paper-figure pipeline defined by ``figures/SPEC.md``. Four panels (page-1 scope
of the fig4 deck; deck ``Fig 4 - Identifying Crit units-1.png`` boxes B/C/E/F —
A/D/G are hand-drawn Illustrator schematics, EXCLUDED per the fig4 contract):

    B  score_curves_hazard_rate.svg   — F1/Accuracy vs. ElasticNet Alpha (binary)
    C  coef_heatmap_hazard_rate.svg   — 32-unit coefficient heatmap
    E  score_curves_contingency.svg   — per-label F1/Accuracy (3-class decode)
    F  coef_heatmap_contingency.svg   — 32-unit coefficient heatmap

Dual use (SPEC rule 7): ``marimo edit figures/fig4_identifying_units.py`` for
interactive work; ``python figures/fig4_identifying_units.py`` runs every cell
top-to-bottom (via ``app.run()``) and lands the 4 SVGs under
``outputs/panels/fig4/``.

Two deliberate, documented deviations from the DS2 source
---------------------------------------------------------
1. **Dead-renderer fix (SPEC ruling 8).** DS2's ``plot_coefs_and_metrics``
   (``:607``) references unbound ``_coefs``/``_hline_chance`` locals (a 2-line
   typo) and would ``NameError`` at call time. The correct form is DS2.1's copy
   (``:614-682``), which binds the parameters; the render helpers below port
   that fixed form (split into two self-contained panels per SPEC rule 1).
2. **Metrics correction.** DS2's contingency render cell (``:766``) reuses
   ``metrics_to_plot=['accuracy']`` left over from the hazard cell, which alone
   would not draw the deck's per-label ``F1 - Label 0/1/2`` lines even though
   the ``average=None`` f1 data exists. Both panels here pass F1 *and* Accuracy;
   the per-stat f1 averaging (binary scalar for hz, per-label vector for the
   3-class contingency decode) is what makes hz render a single ``F1`` line and
   contingency render three — matching the deck.

Scope note: only the deck's exemplar model (``san-4604``) is fitted — the ~100
ElasticNet fits behind these four panels — not DS2's full 10-model loop. Page 1
shows a single model's panels; the multi-model aggregation (deck D/G) is
hand-drawn and excluded. fig4's fits are SPEC rule 4's tolerated inline-compute
exception, memoized via ``transforms.elasticnet_coefficient_paths`` (SPEC
rule 8) so warm reruns are instant.
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
    # expected on this recipe (DS2 ran it the same way). Silence the flood so a
    # headless run's stderr stays readable — it does not affect the fit paths.
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*alpha=0.*")

    from learning_in_context.visualization import paper_style
    from learning_in_context.visualization import transforms

    return (
        np,
        plt,
        TwoSlopeNorm,
        inset_axes,
        paper_style,
        transforms,
    )


@app.cell
def _(paper_style):
    # Style cell: applies the shared theme AND returns render-time constants.
    # Render cells consume these, forcing style-before-render in marimo's DAG
    # (so svg.fonttype='none' is set before any panel is saved).
    paper_style.apply_style()

    # Pinned to the deck's exemplar model (san-4604, lstm, extended_dataset) —
    # the single model whose panels B/C/E/F appear on fig4 deck page 1.
    DATASET = "extended_dataset"
    MODEL = "lstm"
    EXP_ID = "san-4604"

    # Panel sizes from the paper_style vocabulary (SPEC rule 3; per-panel
    # override per ruling 7). Score curves are wide-and-short; heatmaps taller.
    FIGSIZE_SCORE = (paper_style.HALF_WIDTH, 1.7)
    FIGSIZE_HEATMAP = (paper_style.HALF_WIDTH, 2.6)

    return DATASET, MODEL, EXP_ID, FIGSIZE_SCORE, FIGSIZE_HEATMAP


@app.cell
def _(np, plt):
    # Render helper — score curve (ported from DS2.1 plot_coefs_and_metrics's
    # ax1 block, the fixed form; SPEC rule 1: self-contained single panel).
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
def _(np, plt, TwoSlopeNorm, inset_axes):
    # Render helper — coefficient heatmap (ported from DS2.1's ax3 block, fixed
    # form; colorbar labelled "Coefficient Value" per the deck / LIC's
    # critical_units_plots.py, not DS2.1's blank label).
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

    return render_coef_heatmap


# ---------------------------------------------------------------------------
# Transforms (memoized ElasticNet regularization paths — SPEC rules 4 + 8)
# ---------------------------------------------------------------------------
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
def _(np, fit_hz, fit_cont):
    # Vline index = last non-chance alpha (DS2:682 hz offset -3, :744 cont -2).
    def _last_non_chance(metrics, offset):
        acc = np.asarray(metrics["accuracy"])
        return int(np.argmin(np.round(acc, 3))) - offset

    VLINE_HZ = _last_non_chance(fit_hz["metrics"], 3)
    VLINE_CONT = _last_non_chance(fit_cont["metrics"], 2)
    return VLINE_HZ, VLINE_CONT


# ---------------------------------------------------------------------------
# Panel B — hazard-rate score curve
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""## Panels B/C — Hazard rate""")
    return


@app.cell
def _(FIGSIZE_SCORE, VLINE_HZ, fit_hz, paper_style, render_score_curve):
    def _():
        fig = render_score_curve(
            metrics=fit_hz["metrics"],
            C_logspace=fit_hz["C_logspace"],
            metrics_to_plot=["f1", "accuracy"],  # F1 first → deck legend order
            hline_chance=0.5,
            vline_performance=(VLINE_HZ, "accuracy"),
            figsize=FIGSIZE_SCORE,
        )
        paper_style.save_panel(fig, 4, "score_curves_hazard_rate")

    _()
    return


@app.cell
def _(FIGSIZE_HEATMAP, VLINE_HZ, fit_hz, paper_style, render_coef_heatmap):
    def _():
        fig = render_coef_heatmap(
            coefs=fit_hz["coefs"],
            C_logspace=fit_hz["C_logspace"],
            figsize=FIGSIZE_HEATMAP,
            vline_index=VLINE_HZ,
        )
        paper_style.save_panel(fig, 4, "coef_heatmap_hazard_rate")

    _()
    return


# ---------------------------------------------------------------------------
# Panel E — contingency score curve
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""## Panels E/F — Contingency""")
    return


@app.cell
def _(FIGSIZE_SCORE, VLINE_CONT, fit_cont, paper_style, render_score_curve):
    def _():
        fig = render_score_curve(
            metrics=fit_cont["metrics"],
            C_logspace=fit_cont["C_logspace"],
            metrics_to_plot=["f1", "accuracy"],  # f1 is (n_alphas, 3) → 3 lines
            hline_chance=1 / 3,
            vline_performance=(VLINE_CONT, "accuracy"),
            figsize=FIGSIZE_SCORE,
        )
        paper_style.save_panel(fig, 4, "score_curves_contingency")

    _()
    return


@app.cell
def _(FIGSIZE_HEATMAP, VLINE_CONT, fit_cont, paper_style, render_coef_heatmap):
    def _():
        fig = render_coef_heatmap(
            coefs=fit_cont["coefs"],
            C_logspace=fit_cont["C_logspace"],
            figsize=FIGSIZE_HEATMAP,
            vline_index=VLINE_CONT,
        )
        paper_style.save_panel(fig, 4, "coef_heatmap_contingency")

    _()
    return


if __name__ == "__main__":
    app.run()

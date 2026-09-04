# Figures

The marimo scripts that render the paper's figure panels. `fig2_ideal_observer.py` through
`fig7_gates.py` each produce the panels of one paper figure, written to
`figures/panels/fig<N>/<name>.svg`. Every panel is exported self-contained, at its final printed
size, so the composed figure can be assembled from them in a vector editor.

`figures/panels/` is gitignored — the SVGs are build output, regenerated from cached data on demand.

## Contents

| File | What it is |
|---|---|
| `fig2_ideal_observer.py` | Figure 2 — ideal-observer estimate curves and confidence-weighted-choice behaviour panels |
| `fig3_task_results.py` | Figure 3 — task-results CWC grids across participants, RNN, and LSTM |
| `fig4_identifying_units.py` | Figure 4 — ElasticNet score curves and coefficient heatmaps |
| `fig5_unit_activity.py` | Figure 5 — critical-unit activity time-courses and profile scatters |
| `fig6_interventions.py` | Figure 6 — intervention time-courses and summary point plots |
| `fig7_gates.py` | Figure 7 — network-gate point plots and delta-gate scatters |
| `AGENTS.md` | Conventions for agents doing figure work in this directory |
| `CLAUDE.md` | Claude Code adapter — imports `AGENTS.md` |

## Setup

Run once, from the repository root:

```bash
uv sync --extra dev
```

The `dev` extra is what brings in marimo, so it is required even to just open a figure.

## Regenerating panels

```bash
uv run doit panels                          # every figure, skipping what is already current
uv run doit panels:fig4                     # one figure
uv run doit forget panels && uv run doit panels   # force a full re-render
```

`doit` reruns a figure when its script, `paper_style.py`, `transforms.py`, or the upstream cached
data changed — so touching either shared module re-renders every figure.

To run a single figure without `doit`:

```bash
uv run python figures/fig5_unit_activity.py
```

This executes every cell top to bottom and writes that figure's SVGs.

## Working on a figure interactively

```bash
uv run marimo edit figures/fig5_unit_activity.py
```

Every panel is displayed inline as you go. The **Save SVG panels** switch at the top controls
whether rendering also writes to `figures/panels/` — turn it off to iterate on styling without touching the
SVGs on disk. It defaults to on, and headless runs always save.

## Caching

Panels render from memoized transforms cached under `data/cache/fig_transforms`, so a warm run is
quick: it reloads plot-ready tables instead of re-reading the multi-gigabyte state arrays. A cold
run pays the full tier-1 read, plus fig4's ElasticNet fits, and takes appreciably longer.

Two knobs when a cached result looks wrong:

- `LIC_FIG_FORCE_RECOMPUTE=1` in the environment clears the shared cache on import, so that run
  recomputes everything
- `LIC_FIG_CACHE_DIR` points the cache somewhere else — useful for a throwaway cache

Note that joblib keys each cached result on its function's source, so editing a transform (even one
of its comments) already forces a recompute of that transform on the next run.

## Fonts

Figure text is Liberation Sans, vendored in
`src/learning_in_context/visualization/fonts/` and registered at style-apply time, so panels render
identically on any machine. The SVGs keep live, editable text rather than outlined paths — which
means whichever machine opens them for composition needs Liberation Sans installed, or the text
will reflow.

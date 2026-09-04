# Figures — panel authoring

Tier-3 of the figure pipeline: marimo scripts that turn cached tier-2 transforms into the paper's
figure panels. Each script renders one paper figure's panels as individual SVGs under
`figures/panels/fig<N>/`; the composed figure is assembled outside this repo.

## Contents

| Entry | Kind | Holds |
|---|---|---|
| `fig2_ideal_observer.py` | script | figure 2 panels — ideal-observer estimate curves and confidence-weighted-choice behaviour |
| `fig3_task_results.py` | script | figure 3 panels — task-results CWC grids across participants, RNN, and LSTM |
| `fig4_identifying_units.py` | script | figure 4 panels — ElasticNet score curves and coefficient heatmaps |
| `fig5_unit_activity.py` | script | figure 5 panels — critical-unit activity time-courses and profiles |
| `fig6_interventions.py` | script | figure 6 panels — critical-unit intervention time-courses and point plots |
| `fig7_gates.py` | script | figure 7 panels — network-gate point plots and delta-gate scatters |
| `panels/` | dir | rendered output, `fig<N>/<name>.svg` — gitignored and regenerable |
| `__marimo__/` | dir | marimo session state — disposable |
| [README.md](README.md) | file | human-facing guide — setup, regeneration, interactive use, cache knobs |

## Operative rules

- **Panels, not figures** — each SVG is one self-contained panel with its own axes, labels, and legend; never add panel letters, suptitles, or composed grids, because composition happens outside this repo
- **Panel names are stable identifiers** — external compositions link `figures/panels/fig<N>/<name>.svg` by path, so never rename an existing output
- **Live-text SVG at final physical size** — `paper_style.apply_style` sets `svg.fonttype="none"`; every `figsize` is real inches drawn from the `paper_style` size vocabulary (`FULL_WIDTH`, `HALF_WIDTH`, `THIRD_WIDTH`, `PANEL_SQUARE`, `PANEL_TUNING`); panels are placed at 100% and never rescaled, so a panel that does not fit gets a new `figsize` and a re-export
- **The figure font is vendored** in `src/learning_in_context/visualization/fonts/` and registered ahead of the system fonts at style-apply time (the family is whatever `paper_style.FONT_FAMILY` lists), so rendering never depends on system fonts; the machine composing the final figures needs that font installed too, or the live SVG text loses its metrics
- **Exports are deterministic** — `save_panel` pins `svg.hashsalt` per panel and strips the SVG date metadata, and every seaborn CI passes `seed=0` — re-exporting an unchanged panel must be byte-identical, so a diff showing only ids or a date is a regression in this discipline
- **Dual use** — every script runs both under `marimo edit` and headlessly as `python figures/<script>.py`. Each render cell gates `paper_style.save_panel` on `save_svgs.value` and ends on a display expression so the panel shows inline; `tests/test_fig_save_toggle.py` enforces all four halves of that contract statically
- **A new figure script touches four places** — `FIGURE_SCRIPTS` in `tests/test_fig_save_toggle.py`, its own `tests/test_fig<N>_panels.py` contract file, a `PANEL_TASKS` entry in `dodo.py`, and `EXPECTED_SUBTASKS`/`EXPECTED_TARGETS` in `tests/test_dodo_panels.py` — adding a panel to an existing figure still touches the last two
- **Render, never compute** — panels read cached tier-1 artifacts through the memoized transforms in `src/learning_in_context/visualization/transforms.py`; a panel needing data no `dodo.py` task produces gets a new compute task, not an inline computation — fig4's ElasticNet fits are the one tolerated exception, and they are memoized too. Every transform keys on paths, ids, and params only, never a preloaded array, and loads its arrays internally
- **joblib keys on the decorated function's own source** — editing a memoized transform's body, docstring, or even a comment inside it invalidates its cache, while editing a private helper it calls does not — when results look stale, run with `LIC_FIG_FORCE_RECOMPUTE=1` or call `transforms.clear_cache()`
- Editing `paper_style.py` or `transforms.py` marks **every** `panels:*` sub-task stale — both are declared `file_dep` of every figure
- Comments and docstrings in this pipeline are **publish-ready**: self-contained statements about the code, with no references to internal specs, decks, or process artifacts

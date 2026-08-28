<!-- wordsmith: audience=agent function=procedure -->

# SPEC — Paper figure-generation scripts

Build one marimo script per paper figure in this directory (`figures/`), all
importing a shared style module, each exporting its panels as individual SVG
files with live text for external composition in Illustrator. Source code for
every panel is already located — see the mapping table below and
`../../figure-to-code-map.md` (workspace root) for `path:line` detail.

Operator direction (2026-08-28): this spec lives in `figures/SPEC.md`, shipped
with the repo (not the PerAnkh plans tree), so the implementing agent finds it
beside the code. Scope is **page 1 of each figure deck only** — the first pages
hold the figure collections to recreate; later deck pages are out of scope.
Operator direction (2026-08-28): figure panels are composed externally
(Illustrator or similar), never in matplotlib — scripts produce panels, not
composed figures.

## Architecture

| Piece | Path | Role |
|---|---|---|
| Style module | `src/learning_in_context/visualization/paper_style.py` (built) | Single source of truth: theme, palette, fonts, size vocabulary, `apply_style()`, `save_panel()` |
| Transforms module | `src/learning_in_context/visualization/transforms.py` (new) | Shared, memoized figure transformations (tier 2 — see rule 8) |
| Figure scripts | `figures/fig1 … fig7_*.py` (marimo) | One per paper figure; call transforms, style panels, `save_panel()` |
| Panel outputs | `outputs/panels/fig<N>/<panel-name>.svg` | The deliverables Illustrator links to |
| Transform cache | `data/cache/fig_transforms/` (joblib) | Memoized tier-2 results, shared across figure scripts |
| Orchestration | `dodo.py` → new `task_panels` | Regenerates stale panels; depends on artifacts + `paper_style.py` |

Three tiers: **tier 1** `dodo.py` compute → `data/cache/…` (expensive,
model-level); **tier 2** figure transformations → `data/cache/fig_transforms/`
(per-figure reshaping, memoized); **tier 3** panel render → SVG (cheap,
style-only, iterate freely). The DS sources fuse tiers 2–3 inside cells; the
port splits them so styling iteration never re-pays transformation cost.

Rules that make this work:

1. **Panels, not figures.** No panel letters (A/B/C), no suptitles spanning
   panels, no multi-panel layout tuned for adjacency. Each panel is
   self-contained (own axes, labels, legend). Illustrator owns everything
   between panels.
2. **Export SVG with live text.** Set `svg.fonttype: 'none'` in
   `paper_style.py` so text stays editable in Illustrator (matplotlib's default
   outlines it). If PDF output is ever added, use `pdf.fonttype: 42`.
3. **Final physical size in code.** Export at final print size (`figsize` in
   real inches from the size vocabulary: `FULL_WIDTH`, `HALF_WIDTH`, …) with
   fonts at final point size. The Illustrator-side rule is *place at 100%,
   never rescale* — a panel that doesn't fit gets its `figsize` changed in code
   and re-exported. This is the consistency linchpin; state it in a comment in
   `paper_style.py`.
4. **Render only, never compute.** Figure scripts read cached artifacts from
   `data/cache/` (`dodo.py`'s `CACHE_DIR`; produced by the existing compute
   tasks: `task_extract_model_states`, `task_critical_units_*`,
   `task_aggregate_critical_units`, …). If a panel needs data no task
   produces, add a compute task; do not compute inside the figure script.
   Known tolerated exception: fig4's ElasticNet fits run inline in the DS2
   source — port as-is first, route through `task_critical_units_*` later.
5. **Stable panel names.** Illustrator links to
   `outputs/panels/fig<N>/<panel-name>.svg` by path; renaming an output breaks
   the composition. Choose semantic names once (e.g. `fig3/cwc_straight.svg`).
6. **Vendor the font.** The decks use Arial. Ship the `.ttf` in the repo (or
   pick a redistributable substitute) and register it in `paper_style.py` via
   `matplotlib.font_manager`; do not depend on system fonts.
7. **marimo dual use.** Each script is both interactive (`marimo edit`) and
   headless (`python figures/fig_X.py` runs cells top-to-bottom). The last
   cell(s) must call `save_panel()` so a headless run lands files.
8. **Memoize figure transformations with `joblib.Memory`** rooted at
   `data/cache/fig_transforms/`. Transformations (sliding windows, ordering,
   aggregations, CWC groupings — everything between tier-1 artifacts and a
   plot-ready frame) are pure functions decorated with the shared `Memory`
   instance (expose it from `transforms.py`). joblib auto-invalidates on
   function-source or argument change; manual recompute = `memory.clear()`,
   delete the cache dir, or a `FORCE_RECOMPUTE` env knob. **Key on paths +
   params, never on loaded arrays** — signatures take file paths, exp-ids, and
   parameters; the function loads data internally (keeps cache lookup from
   hashing large arrays and keys stable). Do NOT use `mo.persistent_cache`
   (per-cell identity — defeats cross-script sharing).
   Operator direction (2026-08-28): cache lives beside tier-1 artifacts in
   `data/cache/`; refactor scope is **lean** — put a transform in shared
   `transforms.py` only when it demonstrably has (or gains) a second consumer;
   otherwise it stays script-local but still memoized. Promotion later is
   cheap. Known shared-from-day-one: the per-model intervention frames
   (DS6.2 + DS6.4 both build them from `interventions/`) and fig5's
   ordered-change windows (feed both activity panels and profile scatters).

## Panel → source mapping (page-1 scope)

Port from these sources. `DS*` paths are marimo `.py` conversions in
`hmdcpd-analysis/notebooks/` (sibling repo in this workspace); line numbers are
current as of the 2026-08-27 re-derivation. Full per-panel matching evidence:
`../../figure-to-code-map.md`.

| Script | Panels | Port from |
|---|---|---|
| `fig1` | — none. Fig 1 p1 is hand-drawn schematic; no script needed. | — |
| `fig2` | **Pending operator ruling** (see Open questions) | `DS1-Humans-And-Models.py:1180-1296` if ruled in |
| `fig3_task_results.py` | CWC grids: straight-path (x=Grayzone Position, hue=Hazard Rate, cols Participants/RNN/LSTM) + wall-bounce (hue=Contingency) | `DS1-Humans-And-Models.py` — models `:1271-1296`, participants `:1714-1763`, via `hmdcpd.visualization.multi_plot_color_prediction_counts` |
| `fig4_identifying_units.py` | ElasticNet score curves (F1/Accuracy vs Alpha, chance line) + coefficient heatmaps | `DS2-Identifying-Critical-Units.py` — renderer `plot_coefs_and_metrics` `:607`, chance line `:636`, hazard call `:710`. Grayzone variant: `DS2.1-…-Grayzone.py` |
| `fig5_unit_activity.py` | Activity time-courses (Low/High HZ + Cont trials, ordered changes) + activity-profile scatters (Step Size × Activity Decay) | **Retrofit existing** `figures/fig_hazard_rate_activity.py` (time-courses `:334-:762`, scatters `:1055-1057`) — already in this repo; analysis origin `DS3-Neural-Tuning.py:340-653` |
| `fig6_interventions.py` | Intervention time-courses (P(Color Change) vs Timestep, Alpha sweep) + summary point plots (P(Final Color Change) vs Alpha, Type L2L/L2H/H2L/H2H) | `DS4-Interventions.py` — `plot_interventions` `:421`, `plot_interventions_rows` `:522`, hazard `:706+`, contingency `:888+`, point plots: FINAL cells only `:1150-1192`/`:1279-1313`/`:1520-1526` (early-draft duplicates `:820-836`/`:1332-1339` have stale labels — do not port) |
| `fig7_gates.py` | Gate-rescue point plots + TWO gate scatters: per-trial (Blue/Green/Red/Unity) and aggregated unit-mean by Color Entered × Model | `DS6.2-Interventions-and-Gates.py:814-920` (rescue; page 1 needs only the `('i','f')` pair), `DS6.4-Relative-Gate-Activities.py:568-590` (per-trial scatter), aggregated scatter reconstructed from stale backup `notebooks/marimo/DS6.4-Relative-Gate-Activities.py:1150-1195` (no current source exists), `DS4-Interventions.py:1279-1313` (reused cell-unit panel; an earlier `:837` cite was wrong) |

Sibling scripts `fig_contingency_activity.py` / `fig_contingency_activities.py`
in this directory cover Fig 6 p2 material (out of scope) — leave them; they
share code worth reusing for `fig5`/`fig6`.

## Procedure

1. Create `paper_style.py`: `apply_style()` (the one `sns.set_theme` +
   `rcParams` call — lift the current inline block from
   `fig_hazard_rate_activity.py:80-85` as the starting point), palette
   constants, size vocabulary, font registration, `save_panel(fig, fig_no,
   name)` writing SVG to `outputs/panels/`.
2. Create `transforms.py`: the shared `joblib.Memory` instance (rooted per
   rule 8) plus, initially, only the known-shared transforms; everything else
   stays script-local-but-memoized until a second consumer appears.
3. Retrofit `fig_hazard_rate_activity.py` → `fig5_unit_activity.py`: replace
   its inline theme with `apply_style()`, split transformation cells from
   render cells and memoize the transforms (rule 8), strip composition, add
   `save_panel` calls. This validates the style module, the transform cache,
   and the panel conventions against the one already-working script before any
   porting.
4. Port `fig3`, `fig4`, `fig6`, `fig7` from their DS sources (table above).
   Port the plotting code; take data from cached artifacts per rule 4. Where a
   DS notebook computed inline, add the missing compute task to `dodo.py`.
5. Add `task_panels` to `dodo.py`: one sub-task per figure script;
   `file_dep` = its artifacts + `paper_style.py`; `targets` = its SVGs;
   action `python figures/<script>.py`. Verify `doit panels` regenerates
   everything and that touching `paper_style.py` marks all panels stale.
6. Verify each SVG opens with live (selectable) text and matches its deck
   panel in content (not pixel-identical — font rendering will differ from the
   raster originals; that is expected).

## Constraints

- **Never read `.ipynb` files** (workspace policy, hook-enforced). The DS
  `.py` marimo conversions beside them are current (re-derived 2026-08-27) —
  read those.
- `hmdcpd-analysis` is a separate repo: **port** code into this repo (LIC
  is the shipped figure source); do not import `hmdcpd` from LIC scripts.
- Ported scripts consume LIC `data/cache/` artifacts (rule 4). Data-readiness
  (verified 2026-08-28): **all page-1 figures' inputs exist on disk** —
  `model_states/` (3 datasets × 10 LSTM + 10 RNN `.npz`; `ibo/` in
  control/participant only, absent from `extended_dataset`),
  `critical_units/dict_units*.pkl`, `interventions/` (350 files incl.
  gate-frozen rescue variants), `data/raw/` participant data. No compute task
  needs to run first; the five scripts are independent and parallelizable.
  Soft spot: DS1 reads per-model prediction CSVs (`DS1:1125/:1144`) not
  individually verified — fig3's implementer should assert them at startup.
  `tuning_profiles/` is empty but feeds no page-1 figure.
- Run commands from within this repo (it has its own `pyproject.toml`/`uv.lock`).

## Operator rulings (2026-08-28)

1. **Scope: figures 4–7 first.** Fig 2 is excluded (another agent is redoing
   the IBO work) and fig3 is deferred with it — both depend on the DS1
   humans-and-models material under active rework. Revisit once the IBO redo
   lands.
2. **Font: Liberation Sans**, vendored in the repo (metric-compatible Arial
   substitute, freely redistributable). Register it in `paper_style.py` ahead
   of Arial in the stack. Illustrator machines need it installed too.
3. **Review cadence: checkpoint after the fig5 retrofit** — operator blesses
   the conventions on real panels before the fig4/6/7 fan-out.
4. **Git: commit per step** on a work branch/worktree per vcs-rails
   conventions (conventional commits; load `vcr-conventions` before git work).
   Amended 2026-08-28: committing directly to `main` is fine for now (single
   agent working; in-place branch creation is vcs-rails-blocked and a separate
   worktree cannot build a venv while `uv` resolution is broken).

## Operator rulings (2026-08-28, fig5 checkpoint + pre-flight)

5. **Fig5 contingency-scatter recipe confirmed (provisionally).** The deck's
   contingency profile scatters have no surviving source (dead copy-paste
   cells in `fig_contingency_activity.py`); the reconstructed recipe —
   hazard-exemplar units restricted to the 6 contingency models, event =
   wall bounce causing a color change (`targets[...,-2]==1`) — reproduces the
   deck's points exactly and is confirmed for now; the operator will
   investigate further later. Implemented in `transforms.py`
   (`STAT_UNITS["hz_cont"]`, `criterion_mode="bounce_color_change"`).
6. **Condition shorthand is "CT"** (spacing), per
   `paper_style.SHORTENED_CONDITIONS` — the deck's "Cont" labels do NOT
   override this. fig5's current "Cont" titles must be reverted to CT and the
   panels regenerated.
7. **Panel sizing**: shared defaults from the `paper_style` size vocabulary,
   overridden per panel in that panel's render cell when needed (current
   implementation is already this shape).
8. **Fig 4**: fix DS2's dead renderer (2-line `_coefs`/`_hline_chance` bug)
   when porting, matching DS2.1's correct version; DS2.1/grayzone is out of
   scope for page 1.
9. **Fig 6**: correct panel N's title to "All Models Cell Unit Interventions"
   (the deck's "Hidden" is a published copy-paste bug) — a deliberate,
   documented deviation from the deck. Port only the polished final point-plot
   cells (see mapping table).
10. **Fig 7**: reconstruct the aggregated gate scatter from the stale backup
    `notebooks/marimo/DS6.4-Relative-Gate-Activities.py:1150-1195`; the reused
    cell-unit interventions panel ports from `DS4:1279-1313`.
11. **Dependencies**: repair the broken `uv` lock resolution and add `marimo`
    to the declared dependencies (it is currently an ad-hoc venv install);
    fallback if unrepairable within budget: document in `KNOWN-ISSUES.md`.
12. **Unattended completion run**: figs 4–7 run to full completion (SPEC steps
    4–6 plus the fig5 touch-ups above) without intermediate operator
    checkpoints; checkpoint/handoff at the end. Figs 1–3 remain out of scope
    pending their own specification work.

## Design rationale (context, skippable)

- Panels-not-figures and the 100%-placement rule exist because external
  composition breaks font-size consistency the moment any panel is rescaled;
  fixing size in code is the only enforceable point.
- SVG-with-live-text over PDF: Illustrator edits text elements directly;
  matplotlib's default text-to-path conversion would make labels uneditable.
- The DS notebooks (hmdcpd-analysis) remain the exploratory/analysis record;
  this repo's `figures/` is the shipped, citable figure code (`CITATION.cff`
  is here). The mapping table is therefore a migration checklist, not a
  dependency list.
- Provenance chain: deck PDFs → per-page PNGs (`google-drive/paper/png/`) →
  visual label extraction → grep across DS `.py` and LIC — recorded in
  `../../figure-to-code-map.md`, re-verified 2026-08-27 after re-deriving all
  14 DS conversions.

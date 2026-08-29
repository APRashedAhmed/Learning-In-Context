"""Integration tests for the NOT-YET-WRITTEN ``figures/fig6_interventions.py``
(figures/SPEC.md procedure step 4: port from ``DS4-Interventions.py``).

Panel inventory (from ``google-drive/paper/png/Fig 6 - Crit Units Func-1.png``,
the fig6 deck's page 1 -- SPEC scopes each figure to page 1 of its deck)
==========================================================================
Page 1 is a lettered A + G..N layout:

  A - "Schematic of intervention pipeline": hand-drawn placeholder box (plain
      text on a bordered rectangle, no plotted data). Per SPEC's fig1
      treatment ("hand-drawn schematic; no script needed") and the
      fig4/fig5-contract precedent (schematic boxes ruled OUT there), this is
      Illustrator compose-time art -- NOT a generated panel. EXCLUDED.

  Hazard-rate block (G/H, I/J):
    G - "Hidden Unit Intervention: Low to High Hz" / "... High to Low Hz" --
        two side-by-side line subplots (P(Color Change) vs Timestep) sharing
        one legend column between them: "Alpha" (0.0-0.9, viridis ramp),
        "Target" (the opposite-condition reference line, drawn red in-axes),
        "Intervention" (dashed vertical line marking intervention onset).
        A gray dashed vline marks the intervention timestep in each subplot.
        RENDERED -> IN.
    H - "All Models Hidden\nUnit Interventions" point plot: P(Final Color
        Change) vs Alpha (0.0-1.0, x-axis relabeled from the raw 0-10 alpha
        index), hue="Type" with legend entries L2L / H2L / L2H / H2H.
        RENDERED -> IN.
    I - "Cell Unit Intervention: Low to High Hz" / "... High to Low Hz" --
        same layout as G, cell-unit data. RENDERED -> IN.
    J - "All Models Cell\nUnit Interventions" point plot -- same layout as H,
        cell-unit data. RENDERED -> IN.

  Contingency block (K/L, M/N) -- same shapes, "Cont"/"Contingency" data,
  shorter x-axis (8 timesteps vs. 26 for hazard-rate):
    K - "Hidden Unit Intervention: Low to High Cont" / "... High to Low Cont".
        RENDERED -> IN.
    L - "All Models Hidden\nUnit Interventions" point plot (contingency data).
        RENDERED -> IN.
    M - "Cell Unit Intervention: Low to High Cont" / "... High to Low Cont".
        RENDERED -> IN.
    N - "All Models Hidden\nUnit Interventions" point plot **as rendered in
        the deck** -- but this is the published copy-paste bug (SPEC ruling
        9): the underlying data is the CELL-unit contingency point plot
        (built from ``dict_model_pred_dfs_melted_bounce_cont_c``, the same
        "_c" cell-state frame that feeds M), not hidden-unit data. Operator
        ruling 9 (2026-08-28): correct panel N's title to "All Models Cell
        Unit Interventions" in the port -- a DELIBERATE, DOCUMENTED DEVIATION
        from the deck image. This is the ONE title this test asserts must
        differ from the literal deck text.

Panel -> stable semantic SVG name (SPEC rule 5), under ``outputs/panels/fig6/``.
Following the fig5-contract precedent (each deck box = ONE self-contained SVG,
even where the DS source composes two condition-order subplots -- Low-to-X /
High-to-X -- sharing a single legend column via ``plot_interventions_rows``'s
gridspec; SPEC rule 1's "own axes, labels, legend" is read at deck-box
granularity, matching the fig5 checkpoint's already-blessed
``render_timecourse`` shape rather than forcing a further split the deck
itself never draws):

  G -> intervention_timecourse_hz_hidden.svg
  H -> summary_pointplot_hz_hidden.svg
  I -> intervention_timecourse_hz_cell.svg
  J -> summary_pointplot_hz_cell.svg
  K -> intervention_timecourse_ct_hidden.svg
  L -> summary_pointplot_ct_hidden.svg
  M -> intervention_timecourse_ct_cell.svg
  N -> summary_pointplot_ct_cell.svg   (title corrected per ruling 9)

Source mapping (SPEC's fig6 row + figure-to-code-map.md's 2026-08-28 scout
audit "Pre-flight corrections" -- ``DS*`` = hmdcpd-analysis/notebooks/):
  * Time-course renderer: ``plot_interventions_rows`` (``DS4-Interventions.py``
    :522) -- the "Target"/"Intervention" legend titles are literal strings in
    this function (:597 ``ref_title = f'Target'``; the dashed-vline legend at
    :608 is literally ``'Intervention'``), confirmed against the deck legend
    crop (Alpha / Target / Intervention, exactly as rendered) -- this is NOT
    the sibling singular-panel ``plot_interventions`` (:421), whose reference
    legend instead computes a "{Cond} {Hz|Cont}" title and is used only for
    the bounce/control-trial cells that are out of scope for page 1.
  * G data/call: ``:1137-1147`` (hz, hidden -- ``dict_model_pred_dfs_melted_straight_hz_h``,
    ``title="Hidden Unit Intervention:\n"``).
  * I data/call: ``:1265-1276`` (hz, cell -- ``dict_model_pred_dfs_melted_straight_hz_c``,
    ``title="Cell Unit Intervention:\n"``).
  * K data/call: ``:1397-1408`` (cont, hidden -- ``dict_model_pred_dfs_melted_bounce_cont_h``,
    ``title="Hidden Unit Intervention:\n"``).
  * M data/call: ``:1497-1508`` (cont, cell -- ``dict_model_pred_dfs_melted_bounce_cont_c``,
    ``title="Cell Unit Intervention:\n"``).
  * Summary point plots each have TWO live source cells per the scout audit;
    port the polished FINALS only (the early drafts at ``:820-836``/``:1332-1339``
    use stale "Intervention Strength (Alpha)"/"Final Color Change Probability"
    axis wording -- do NOT port that wording):
      H final: ``:1176-1216`` -- ``plt.title('All Models Hidden\nUnit Interventions')``.
      J final: ``:1278-1318`` -- ``plt.title('All Models Cell\nUnit Interventions')``.
      L final: ``:1410-1450`` -- ``plt.title('All Models Hidden\nUnit Interventions')``
          (correctly "Hidden" as rendered -- this IS hidden-unit data; no
          deviation here).
      N final: ``:1510-1551`` -- ``plt.title('All Models Hidden\nUnit Interventions')``
          in DS4 (and the deck), but built from the cell-state contingency
          frame (``dict_model_pred_dfs_melted_bounce_cont_c``, the same input
          M uses) -- ruling 9 corrects this to "Cell" in the port.
  * All four summary point plots share: ``x='Alpha'`` (raw index rescaled to
    0.0-1.0 via dynamic xticks, not the old ``FuncFormatter(x * 0.1)`` used by
    the early-draft cells), ``y='Value'`` -> ylabel ``'P(Final Color Change)'``,
    ``hue='Type'`` with palette order ``['L2L', 'L2H', 'H2L', 'H2H']`` and
    ``Type`` values computed as e.g. ``'L' if <cond>=='Low' else 'H'`` + ``'2'``
    + ``'L' if Centroid==0 else 'H'`` (so the rendered legend/data entries are
    literally ``L2L``/``L2H``/``H2L``/``H2H`` -- this test asserts all four are
    present, not the palette's declared order, since seaborn's actual legend
    order follows the data's hue order which this test does not independently
    re-derive).

Judgment calls for verifier attention
--------------------------------------
  * HZ/CT casing: the deck's literal title text reads "...Hz" / "...Cont"
    (mixed case, straight from DS4's ``stat_short = 'Hz' if 'hazard' in
    stat.lower() else 'Cont'``). SPEC ruling 6 (2026-08-28, fig5 checkpoint)
    states "Condition shorthand is 'CT' ... the deck's 'Cont' labels do NOT
    override this," and the ALREADY-IMPLEMENTED ``fig5_unit_activity.py``
    applies this as a general paper_style convention -- it passes literal
    ``"HZ"`` (uppercase) for hazard-rate titles too, not just "CT" for
    contingency (see ``fig5_unit_activity.py:271,365,458``). Ruling 6 was
    worded under the "fig5 checkpoint" heading, not explicitly re-stated for
    fig6, but ``paper_style.SHORTENED_CONDITIONS`` is a shared module-level
    constant, not fig5-scoped, and this test treats the fig5 precedent as
    binding for consistency across the paper's panels. This test therefore
    asserts uppercase ``"HZ"``/``"CT"`` in fig6's time-course titles (not the
    deck's literal "Hz"/"Cont" mixed case) -- flagged here as a judgment call
    the verifier should confirm rather than silently accept.
  * Ruling 9 is scoped narrowly to panel N's *title* text ("correct panel N's
    title to 'All Models Cell Unit Interventions'"). This test does NOT
    require the underlying implementation to literally reuse M's cached
    dataframe/transform for N (an implementation-detail choice for whichever
    agent ports it) -- only that the rendered SVG's title text says "Cell",
    not "Hidden".
  * The G/I/K/M time-course panels are NOT split into per-condition
    (Low-to-High vs. High-to-Low) SVGs even though SPEC rule 1 nominally
    reads as one-axes-per-panel. This mirrors the fig5-contract's already-
    blessed ``render_timecourse`` shape (1x2 subplots, one shared legend
    column) and is treated as the settled precedent rather than re-litigated
    here; a verifier who disagrees should flag it explicitly, since it
    affects EXPECTED_PANELS' count and names.
  * "All Models" scope: ``data/cache/interventions/`` (SPEC's data-readiness
    note) holds LSTM-only per-model intervention artifacts (10
    ``san-*`` models under ``interventions/lstm/``) -- no ``interventions/rnn/``
    directory exists. Hidden/cell-state interventions are LSTM-specific (RNN
    has no separate cell state), so "All Models" here plausibly means "all
    LSTM models," consistent with the cached data actually present. This test
    does not assert a specific model count (e.g. "10 models") since that is a
    data-fidelity claim, not a panel-existence/content claim.
  * fig5's shared-transform naming precedent (SPEC rule 8's "known
    shared-from-day-one": "the per-model intervention frames (DS6.2 + DS6.4
    both build them from interventions/)") is exercised indirectly here via
    ``TestTransformMemoization`` (cache-dir populated after a headless run),
    mirroring fig4/fig5's contract -- this test does not pin the shared
    transform's exact name/signature (that is ``transforms.py``'s contract,
    covered by ``tests/test_fig_transforms.py``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG6_SCRIPT = REPO_ROOT / "figures" / "fig6_interventions.py"
PANELS_DIR = REPO_ROOT / "outputs" / "panels" / "fig6"

EXPECTED_PANELS = [
    "intervention_timecourse_hz_hidden.svg",   # G
    "summary_pointplot_hz_hidden.svg",         # H
    "intervention_timecourse_hz_cell.svg",     # I
    "summary_pointplot_hz_cell.svg",           # J
    "intervention_timecourse_ct_hidden.svg",   # K
    "summary_pointplot_ct_hidden.svg",         # L
    "intervention_timecourse_ct_cell.svg",     # M
    "summary_pointplot_ct_cell.svg",           # N (ruling 9: title says "Cell")
]

# Time-course panels: line subplots, P(Color Change) vs Timestep, Alpha ramp.
TIMECOURSE_PANELS = {
    "intervention_timecourse_hz_hidden.svg": "HZ",
    "intervention_timecourse_hz_cell.svg": "HZ",
    "intervention_timecourse_ct_hidden.svg": "CT",
    "intervention_timecourse_ct_cell.svg": "CT",
}

# Summary point plots: P(Final Color Change) vs Alpha, hue=Type.
POINTPLOT_TITLES = {
    "summary_pointplot_hz_hidden.svg": "All Models Hidden",   # H
    "summary_pointplot_hz_cell.svg": "All Models Cell",       # J
    "summary_pointplot_ct_hidden.svg": "All Models Hidden",   # L
    # N: ruling 9 -- the deck/DS4 literally say "Hidden" here (a published
    # copy-paste bug plotting cell-unit contingency data); the port must say
    # "Cell". This is the one title this test requires to DIFFER from the
    # deck's literal text.
    "summary_pointplot_ct_cell.svg": "All Models Cell",       # N
}

pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def fig6_run(tmp_path_factory):
    """Run figures/fig6_interventions.py headlessly, once per test session.

    Overrides the transforms cache-dir seam (``LIC_FIG_CACHE_DIR``, see
    tests/test_fig_transforms.py, tests/test_fig4_panels.py,
    tests/test_fig5_panels.py) to an isolated tmp dir so this test proves the
    memoization contract without depending on -- or polluting -- the real
    ``data/cache/fig_transforms``. Panel SVGs, by contrast, are written to the
    REAL ``outputs/panels/fig6/`` -- that is the actual deliverable this
    script exists to produce (SPEC architecture table), not a test fixture.

    fig6 is render-only (SPEC rule 4 -- unlike fig4, it has no tolerated
    inline-compute exception; all its inputs already exist under
    ``data/cache/interventions/``), so this uses fig5's 600s timeout rather
    than fig4's 1800s.
    """
    cache_dir = tmp_path_factory.mktemp("fig_transforms_cache")
    env = dict(os.environ)
    env["LIC_FIG_CACHE_DIR"] = str(cache_dir)

    result = subprocess.run(
        [sys.executable, str(FIG6_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result, cache_dir


class TestHeadlessRun:
    def test_script_exists(self):
        # Fails clearly ("script not found") ahead of the subprocess call,
        # rather than as an opaque non-zero-exit / "No such file" surprise.
        assert FIG6_SCRIPT.exists(), f"not yet written: {FIG6_SCRIPT}"

    def test_exits_zero(self, fig6_run):
        result, _ = fig6_run
        assert result.returncode == 0, (
            "figures/fig6_interventions.py exited "
            f"{result.returncode}\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )


class TestPanelOutputs:
    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_writes_expected_panel(self, fig6_run, panel_name):
        result, _ = fig6_run
        if result.returncode != 0:
            pytest.fail(f"headless run failed:\n{result.stderr}")
        panel_path = PANELS_DIR / panel_name
        assert panel_path.exists(), f"expected panel not written: {panel_path}"

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_has_live_text_elements(self, fig6_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "<text" in content, f"{panel_name} has no live <text> elements"

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_is_a_single_self_contained_svg(self, fig6_run, panel_name):
        # Cheap proxy for "panel, not a composed multi-panel grid" (SPEC
        # rule 1) -- exactly one <svg> root element in the document. Note
        # this does NOT forbid multiple <g>/<axes> groups inside that one
        # root -- G/I/K/M's shared-legend 1x2 subplot layout (fig5 precedent,
        # see module docstring) still exports as a single <svg> document.
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert content.count("<svg") == 1, (
            f"{panel_name} does not look like a single self-contained panel "
            f"(found {content.count('<svg')} <svg> tags)"
        )

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_is_non_trivial_size(self, fig6_run, panel_name):
        # Cheap proxy against an accidentally-blank/near-empty export.
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        assert panel_path.stat().st_size > 2000, (
            f"{panel_name} is suspiciously small "
            f"({panel_path.stat().st_size} bytes) for a rendered panel"
        )


class TestTimecourseContent:
    """Deck-verified text for panels G/I/K/M (intervention time-courses).

    Shared axis/legend vocabulary: y-axis "P(Color Change)", x-axis
    "Timestep", an "Alpha" legend (viridis ramp over 0.0-0.9), a "Target"
    legend entry, and an "Intervention" legend entry (both literal strings in
    ``plot_interventions_rows`` -- see module docstring's source mapping).
    Titles read "<Hidden|Cell> Unit Intervention:" plus "Low to High <HZ|CT>"
    / "High to Low <HZ|CT>" for the two composed condition-order subplots.
    """

    @pytest.mark.parametrize("panel_name", list(TIMECOURSE_PANELS))
    def test_has_shared_axis_and_legend_text(self, fig6_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "P(Color Change)" in content, f"{panel_name} missing y-axis label"
        assert "Timestep" in content, f"{panel_name} missing x-axis label"
        assert "Alpha" in content, f"{panel_name} missing 'Alpha' legend title"
        assert "Target" in content, f"{panel_name} missing 'Target' legend entry"
        assert "Intervention" in content, (
            f"{panel_name} missing 'Intervention' legend entry"
        )

    @pytest.mark.parametrize("panel_name", list(TIMECOURSE_PANELS))
    def test_has_low_to_high_and_high_to_low_titles(self, fig6_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        stat_short = TIMECOURSE_PANELS[panel_name]
        assert f"Low to High {stat_short}" in content, (
            f"{panel_name} missing 'Low to High {stat_short}' subplot title"
        )
        assert f"High to Low {stat_short}" in content, (
            f"{panel_name} missing 'High to Low {stat_short}' subplot title"
        )

    def test_hidden_panels_say_hidden_unit_intervention(self, fig6_run):
        for panel_name in ("intervention_timecourse_hz_hidden.svg", "intervention_timecourse_ct_hidden.svg"):
            panel_path = PANELS_DIR / panel_name
            if not panel_path.exists():
                pytest.skip("panel not written; see test_writes_expected_panel")
            content = panel_path.read_text()
            assert "Hidden Unit Intervention" in content, (
                f"{panel_name} missing 'Hidden Unit Intervention' title text"
            )

    def test_cell_panels_say_cell_unit_intervention(self, fig6_run):
        for panel_name in ("intervention_timecourse_hz_cell.svg", "intervention_timecourse_ct_cell.svg"):
            panel_path = PANELS_DIR / panel_name
            if not panel_path.exists():
                pytest.skip("panel not written; see test_writes_expected_panel")
            content = panel_path.read_text()
            assert "Cell Unit Intervention" in content, (
                f"{panel_name} missing 'Cell Unit Intervention' title text"
            )


class TestSummaryPointPlotContent:
    """Deck-verified text for panels H/J/L/N (summary point plots).

    Shared axis/legend vocabulary: x-axis "Alpha", y-axis "P(Final Color
    Change)", a "Type" hue legend with entries L2L/H2L/L2H/H2H. Titles read
    "All Models <Hidden|Cell>\\nUnit Interventions" -- EXCEPT panel N, whose
    deck/DS4 title literally says "Hidden" (a published copy-paste bug over
    cell-unit contingency data) and which SPEC ruling 9 requires be corrected
    to "Cell" in the port. That correction is this module's one deliberate
    deck-text deviation (see POINTPLOT_TITLES and the module docstring).
    """

    @pytest.mark.parametrize("panel_name", list(POINTPLOT_TITLES))
    def test_has_shared_axis_and_legend_text(self, fig6_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "Alpha" in content, f"{panel_name} missing x-axis label"
        assert "P(Final Color Change)" in content, f"{panel_name} missing y-axis label"
        assert "Type" in content, f"{panel_name} missing 'Type' legend title"
        for label in ("L2L", "H2L", "L2H", "H2H"):
            assert label in content, f"{panel_name} missing legend entry {label!r}"

    @pytest.mark.parametrize("panel_name,expected_prefix", list(POINTPLOT_TITLES.items()))
    def test_title_text(self, fig6_run, panel_name, expected_prefix):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert expected_prefix in content, (
            f"{panel_name} missing title text {expected_prefix!r}"
        )
        assert "Unit Interventions" in content, (
            f"{panel_name} missing 'Unit Interventions' title text"
        )

    def test_panel_n_title_deviates_from_deck_and_says_cell(self, fig6_run):
        # SPEC ruling 9, restated: the deck literally renders "All Models
        # Hidden\nUnit Interventions" for panel N even though the underlying
        # data is cell-unit contingency data (the same frame M plots). The
        # port must NOT reproduce the deck's literal "Hidden" text here.
        panel_path = PANELS_DIR / "summary_pointplot_ct_cell.svg"
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "All Models Cell" in content, (
            "panel N (summary_pointplot_ct_cell.svg) must read "
            "'All Models Cell Unit Interventions' per SPEC ruling 9, "
            "correcting the deck's published 'Hidden' copy-paste bug"
        )


class TestTransformMemoization:
    def test_cache_dir_populated_after_headless_run(self, fig6_run):
        result, cache_dir = fig6_run
        if result.returncode != 0:
            pytest.fail(f"headless run failed:\n{result.stderr}")
        assert cache_dir.exists(), f"transform cache dir was never created: {cache_dir}"
        cached_files = [p for p in cache_dir.rglob("*") if p.is_file()]
        assert cached_files, (
            f"transform cache dir {cache_dir} has no cached results -- fig6's "
            "per-model intervention frames are SPEC rule 8's explicit "
            "known-shared-from-day-one transform and must be memoized via "
            "transforms.py's joblib Memory"
        )

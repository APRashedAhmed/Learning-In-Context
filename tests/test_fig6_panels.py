"""Integration tests for ``figures/fig6_interventions.py``.

Panel inventory
===============
Figure 6 is a lettered A + G..N layout:

  A - "Schematic of intervention pipeline": a hand-drawn schematic (plain text
      on a bordered rectangle, no plotted data), composed externally. No script
      generates it. EXCLUDED.

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
    N - "All Models Hidden\nUnit Interventions" point plot **as published** --
        but that title is a copy-paste bug: the underlying data is the
        CELL-unit contingency point plot, built from the same cell-state
        contingency frame that feeds M, not hidden-unit data. The port
        corrects panel N's title to "All Models Cell Unit Interventions" -- a
        deliberate, documented deviation from the published image, and the ONE
        title this test asserts must differ from it.

Panel -> stable semantic SVG name, under ``figures/panels/fig6/``. Each panel
of the composed figure is ONE self-contained SVG, even where the source
composes two condition-order subplots -- Low-to-X / High-to-X -- sharing a
single legend column via a gridspec. "Own axes, labels, legend" is read at
composed-panel granularity, matching fig5's ``render_timecourse`` shape rather
than forcing a further split the published figure never draws:

  G -> intervention_timecourse_hz_hidden.svg
  H -> summary_pointplot_hz_hidden.svg
  I -> intervention_timecourse_hz_cell.svg
  J -> summary_pointplot_hz_cell.svg
  K -> intervention_timecourse_ct_hidden.svg
  L -> summary_pointplot_ct_hidden.svg
  M -> intervention_timecourse_ct_cell.svg
  N -> summary_pointplot_ct_cell.svg   (title corrected, see above)

Provenance
----------
The panels are ported from the exploratory analysis notebooks in the sibling
``hmdcpd-analysis`` repo:

  * Time-course renderer: the source's multi-row intervention renderer, whose
    "Target"/"Intervention" legend titles are literal strings in that function
    (matching the published legend column: Alpha / Target / Intervention). This
    is NOT its sibling single-panel renderer, whose reference legend instead
    computes a "{Cond} {Hz|Cont}" title and serves the bounce/control-trial
    cells the paper does not show.
  * G/I: hazard-rate hidden/cell straight-trial frames, titled "Hidden Unit
    Intervention:" / "Cell Unit Intervention:".
  * K/M: contingency hidden/cell bounce-trial frames, same titles.
  * Summary point plots: the source has both early drafts and polished final
    cells; only the finals are ported. The drafts' stale "Intervention Strength
    (Alpha)" / "Final Color Change Probability" axis wording is deliberately
    NOT carried over.
      H: "All Models Hidden\nUnit Interventions".
      J: "All Models Cell\nUnit Interventions".
      L: "All Models Hidden\nUnit Interventions" -- correctly "Hidden"; this IS
         hidden-unit data, so no deviation.
      N: published as "All Models Hidden\nUnit Interventions" but built from
         the cell-state contingency frame (the same input M uses); the port
         corrects it to "Cell".
  * All four summary point plots share: ``x='Alpha'`` (raw index rescaled to
    0.0-1.0 via dynamic xticks, not the early drafts' ``FuncFormatter(x*0.1)``),
    ``y='Value'`` -> ylabel ``'P(Final Color Change)'``, ``hue='Type'`` with
    palette order ``['L2L', 'L2H', 'H2L', 'H2H']`` and ``Type`` values computed
    as e.g. ``'L' if <cond>=='Low' else 'H'`` + ``'2'`` + ``'L' if Centroid==0
    else 'H'`` (so the rendered legend/data entries are literally
    ``L2L``/``L2H``/``H2L``/``H2H``). This test asserts all four are present,
    not the palette's declared order, since seaborn's legend order follows the
    data's hue order, which this test does not independently re-derive.

Scope notes
-----------
  * HZ/CT casing: the published title text reads "...Hz" / "...Cont" (mixed
    case, straight from the source). The paper's condition shorthand is
    uppercase ``HZ``/``CT``, defined once in
    ``paper_style.SHORTENED_CONDITIONS`` and applied across every figure
    script, so this test asserts uppercase in fig6's time-course titles.
  * The panel-N title correction is scoped to the *title* text. This test does
    not require the implementation to literally reuse M's cached
    dataframe/transform for N -- only that the rendered SVG's title says
    "Cell", not "Hidden".
  * The G/I/K/M time-course panels are NOT split into per-condition
    (Low-to-High vs. High-to-Low) SVGs. One SVG per panel of the composed
    figure is the granularity, matching fig5's ``render_timecourse`` shape
    (1x2 subplots, one shared legend column).
  * "All Models" scope: ``data/cache/interventions/`` holds LSTM-only per-model
    intervention artifacts (10 ``san-*`` models under ``interventions/lstm/``);
    there is no ``interventions/rnn/`` directory. Hidden/cell-state
    interventions are LSTM-specific (a plain RNN has no separate cell state),
    so "All Models" means all LSTM models. This test does not assert a specific
    model count, which is a data-fidelity claim rather than a panel-content one.
  * The per-model intervention frames are shared with fig7, so they live in
    ``transforms.py`` as one memoized transform. That sharing is exercised
    indirectly here via ``TestTransformMemoization`` (cache dir populated after
    a headless run); the transform's name and signature are pinned by
    ``tests/test_fig_transforms.py``, not here.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG6_SCRIPT = REPO_ROOT / "figures" / "fig6_interventions.py"
PANELS_DIR = REPO_ROOT / "figures" / "panels" / "fig6"

EXPECTED_PANELS = [
    "intervention_timecourse_hz_hidden.svg",   # G
    "summary_pointplot_hz_hidden.svg",         # H
    "intervention_timecourse_hz_cell.svg",     # I
    "summary_pointplot_hz_cell.svg",           # J
    "intervention_timecourse_ct_hidden.svg",   # K
    "summary_pointplot_ct_hidden.svg",         # L
    "intervention_timecourse_ct_cell.svg",     # M
    "summary_pointplot_ct_cell.svg",           # N (title corrected to "Cell")
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
    # N: the published panel and its source literally say "Hidden" here (a
    # copy-paste bug over cell-unit contingency data); the port must say
    # "Cell". This is the one title this test requires to DIFFER from the
    # published text.
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
    REAL ``figures/panels/fig6/`` -- that is the actual deliverable this
    script exists to produce, not a test fixture.

    fig6 is render-only -- unlike fig4 it has no inline compute; all its
    inputs already exist under
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
        assert FIG6_SCRIPT.exists(), f"figure script not found: {FIG6_SCRIPT}"

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
        # Cheap proxy for "panel, not a composed multi-panel grid" -- exactly
        # one <svg> root element in the document. Note
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
    """Published text for panels G/I/K/M (intervention time-courses).

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
    """Published text for panels H/J/L/N (summary point plots).

    Shared axis/legend vocabulary: x-axis "Alpha", y-axis "P(Final Color
    Change)", a "Type" hue legend with entries L2L/H2L/L2H/H2H. Titles read
    "All Models <Hidden|Cell>\\nUnit Interventions" -- EXCEPT panel N, whose
    published title says "Hidden" (a copy-paste bug over cell-unit contingency
    data) and which the port corrects to "Cell". That correction is this
    module's one deliberate deviation from the published text (see
    POINTPLOT_TITLES and the module docstring).
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

    def test_contingency_cell_pointplot_title_says_cell(self, fig6_run):
        # The published panel N renders "All Models Hidden\nUnit
        # Interventions" even though the underlying data is cell-unit
        # contingency data (the same frame M plots). The port must NOT
        # reproduce that "Hidden" text here.
        panel_path = PANELS_DIR / "summary_pointplot_ct_cell.svg"
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "All Models Cell" in content, (
            "panel N (summary_pointplot_ct_cell.svg) must read "
            "'All Models Cell Unit Interventions', correcting the "
            "published 'Hidden' copy-paste bug"
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
            "per-model intervention frames are shared with fig7 and must be "
            "memoized via transforms.py's shared joblib Memory"
        )

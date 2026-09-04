"""Integration tests for ``figures/fig7_gates.py``.

Panel inventory
===============
Figure 7 lays out two lettered blocks, A (top-left) and B (top-right), plus two
unlettered scatter plots directly below them:

  A - "Schematic for gate 'rescue' experiments": a hand-drawn schematic (plain
      text on a bordered rectangle, no plotted data), composed externally. No
      script generates it. EXCLUDED.
  B, left  - "All Models Cell Unit Interventions" point plot. x "Alpha"
      (0.0-1.0), y "P(Final Color Change)" (~0.2-0.5). This is the REUSED
      cell-unit interventions panel -- same rendered content as fig6's "All
      Models Cell Unit Interventions" panel, but exported into fig7's own
      ``figures/panels/fig7/`` namespace, since panel paths are per-figure. The
      Type legend (L2L/H2L/L2H/H2H, flare-palette point-and-line series) is too
      large to sit in the panel, so it is stripped here and exported as its own
      panel (see "Type legend"). RENDERED -> IN.
  B, right - "All Models (i, f) Gate\\nRescue Interventions" point plot. Same
      point-plot shape as B-left (Alpha vs P(Final Color Change)) but data
      comes from gate-frozen ("rescue") interventions rather than cell-unit
      interventions, and the title names the specific gate pair rendered:
      ``(i, f)`` -- the only gate pair the paper shows, though the source loops
      over every ``combinations(gate_order, 2)`` pair. Its Type legend is
      likewise stripped (shared standalone legend), and it additionally drops
      its y-axis label -- the two point plots need not share an identical
      numeric range, so B-left's y-label reads for both. RENDERED -> IN.
  Type legend - the L2L/L2H/H2L/H2H Type legend shared by the two point plots,
      rendered ALONE on its own figure (no data axes) so it can be placed
      independently when the figure is composed. RENDERED -> IN.
  Bottom-left  - untitled scatter. x "Delta Forget Gate Activity"
      (~-0.3..0.3), y "Delta Input Gate Activity" (~-0.3..0.3), legend:
      "Blue" / "Green" / "Red" (color_entered categories) + "Unity" (the
      dashed diagonal reference line). One point per (critical unit x
      color_entered) for a single exemplar model -- NOT per trial (see
      "Scope notes"). RENDERED -> IN.
  Bottom-right - "Delta (High Hz - Low Hz) Forget vs Input\\n Unit-Mean by
      Color Entered x Model" scatter. x "Delta Forget Gate Activity
      (unit-mean)", y "Delta Input Gate Activity (unit-mean)", same
      Blue/Green/Red/Unity legend, aggregated (mean) across each model's
      critical units within each (color_entered, model) pair. RENDERED -> IN.

Panel -> stable semantic SVG name, under ``figures/panels/fig7/``. These names
are the contract: the composed figure links to these paths, so an existing
output is never renamed.
  B, left      -> cell_unit_interventions_all_models.svg
  B, right     -> gate_rescue_input_forget.svg
  Type legend  -> interventions_legend.svg
  bottom-left  -> gate_scatter_delta_forget_input.svg
  bottom-right -> gate_scatter_delta_forget_input_unit_mean.svg

Provenance
----------
The panels are ported from the exploratory analysis notebooks in the sibling
``hmdcpd-analysis`` repo:

  * cell_unit_interventions_all_models.svg -- the source's cell-unit
    interventions cell (not its hidden-unit early draft). Title ``'All Models
    Cell\\nUnit Interventions'``, flare palette over
    ``['L2L','L2H','H2L','H2H']``, ``sns.pointplot(df, x='Alpha', y='Value',
    hue='Type', ...)`` where ``df`` is the concatenated per-model straight-trial
    cell-state frame filtered to ``Timestep == 24``.
  * gate_rescue_input_forget.svg -- the source's gate-rescue cell, which loops
    over gate pairs and titles each ``f'All Models {gate} Gate\\nRescue
    Interventions'``; only the ``('i', 'f')`` iteration is in scope. Same
    point-plot shape/palette as the cell-unit panel; y is still "P(Final Color
    Change)" even though gate freezing is applied at every timestep, not only
    the final one (the source builds the full melted frame then filters
    ``Timestep == 24``).
  * gate_scatter_delta_forget_input.svg -- the source's single-model gate
    scatter. Its frame is a SINGLE-MODEL table: critical-unit gate deltas
    (High Hz minus Low Hz rescue-intervention activity, signed so the
    direction always reads "toward the target hazard rate") averaged per
    ``(color_entered, unit_idx)`` for one exemplar model. Axis labels are the
    title-cased gate names, "Forget" / "Input". The source's title call is
    commented out and the published panel shows no title -- consistent, so
    none is asserted.
  * gate_scatter_delta_forget_input_unit_mean.svg -- reconstructed from an
    older copy of the source notebook; no live cell for it survives, though its
    input frame (the per-model, per-unit table -- every model's copy of the
    single-model frame concatenated) is still live in the current one. The
    reconstruction is
    ``frame.groupby(['color_entered','model'], as_index=False)[['i','f','g','o']].mean()``
    plotted as the same x='f'/y='i' scatter, with an explicit title (unlike the
    single-model cell) and "(unit-mean)" appended to both axis labels. That
    reduces 480 points to 30 (3 colors x 10 models).

Scope notes
-----------
  * "Per-trial" wording: the bottom-left panel is sometimes described as a
    per-trial gate scatter, but its data is per (critical unit x color_entered)
    for ONE exemplar model -- trial identity is averaged out by the source's
    ``groupby('color_entered')[...].mean()`` before the melt/merge back to one
    row per unit. The panel name and these content assertions follow the data
    (unit-level, single model); the published point count (dozens, consistent
    with ~32 units x 3 colors, not hundreds of trials) agrees.
  * Which exemplar model backs the single-model scatter is not asserted: no
    model name is drawn into that panel. The figure script picks a concrete,
    cache-backed exemplar (``san-4604``) deliberately.
  * Gate-pair title formatting: interpolating the raw Python tuple
    ``('i', 'f')`` into the title f-string would render its repr -- quoted
    elements, ``"All Models ('i', 'f') Gate\\nRescue Interventions"``. The
    published title is unquoted, ``"(i, f)"``, so the script formats the pair
    explicitly and this test asserts the unquoted form.
  * Fig7 reads the gate-frozen rescue intervention caches
    (``data/cache/interventions/lstm/<exp_id>/hz-cell-fi-gates-frozen-
    centroid-interventions-11-all-states-alphas.npz``, present for all 10 LSTM
    models) and ``critical_units/dict_units*.pkl`` for the unit-mean
    aggregation. Those are render-only reads; this test does not re-verify the
    cache contents, only that the headless run succeeds and the panels it
    should produce exist with the expected text.
  * The per-model intervention frames are shared with fig6, so they live in
    ``transforms.py``. ``TestTransformMemoization`` below only requires SOME
    cached artifact to land in the isolated ``LIC_FIG_CACHE_DIR`` after a run;
    the transform's name and signature are pinned by
    ``tests/test_fig_transforms.py``, not here.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG7_SCRIPT = REPO_ROOT / "figures" / "fig7_gates.py"
PANELS_DIR = REPO_ROOT / "figures" / "panels" / "fig7"

EXPECTED_PANELS = [
    "cell_unit_interventions_all_models.svg",       # B, left (reused from fig6)
    "gate_rescue_input_forget.svg",                 # B, right ((i, f) pair)
    "interventions_legend.svg",                     # standalone Type legend
    "gate_scatter_delta_forget_input.svg",           # bottom-left (single model)
    "gate_scatter_delta_forget_input_unit_mean.svg",  # bottom-right (aggregated)
]

# Panels sharing the "Alpha vs P(Final Color Change)" point-plot shape
# (Type: L2L/H2L/L2H/H2H legend).
POINT_PLOT_PANELS = [
    "cell_unit_interventions_all_models.svg",
    "gate_rescue_input_forget.svg",
]

# Panels sharing the "Delta Forget vs Delta Input" scatter shape
# (Blue/Green/Red/Unity legend).
GATE_SCATTER_PANELS = [
    "gate_scatter_delta_forget_input.svg",
    "gate_scatter_delta_forget_input_unit_mean.svg",
]

pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def fig7_run(tmp_path_factory):
    """Run figures/fig7_gates.py headlessly, once per test session.

    Overrides the transforms cache-dir seam (``LIC_FIG_CACHE_DIR``, see
    tests/test_fig_transforms.py, tests/test_fig4_panels.py, and
    tests/test_fig5_panels.py) to an isolated tmp dir so this test proves the
    memoization contract without depending on -- or polluting --
    the real ``data/cache/fig_transforms``. Panel SVGs, by contrast, are
    written to the REAL ``figures/panels/fig7/`` -- that is the actual
    deliverable this script exists to produce, not
    a test fixture.
    """
    cache_dir = tmp_path_factory.mktemp("fig_transforms_cache")
    env = dict(os.environ)
    env["LIC_FIG_CACHE_DIR"] = str(cache_dir)

    result = subprocess.run(
        [sys.executable, str(FIG7_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    return result, cache_dir


class TestHeadlessRun:
    def test_script_exists(self):
        # Fails clearly ("script not found") ahead of the subprocess call,
        # rather than as an opaque non-zero-exit / "No such file" surprise.
        assert FIG7_SCRIPT.exists(), f"figure script not found: {FIG7_SCRIPT}"

    def test_exits_zero(self, fig7_run):
        result, _ = fig7_run
        assert result.returncode == 0, (
            "figures/fig7_gates.py exited "
            f"{result.returncode}\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )


class TestPanelOutputs:
    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_writes_expected_panel(self, fig7_run, panel_name):
        result, _ = fig7_run
        if result.returncode != 0:
            pytest.fail(f"headless run failed:\n{result.stderr}")
        panel_path = PANELS_DIR / panel_name
        assert panel_path.exists(), f"expected panel not written: {panel_path}"

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_has_live_text_elements(self, fig7_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "<text" in content, f"{panel_name} has no live <text> elements"

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_is_a_single_self_contained_svg(self, fig7_run, panel_name):
        # Cheap proxy for "panel, not a composed multi-panel grid" -- exactly
        # one <svg> root element in the document.
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert content.count("<svg") == 1, (
            f"{panel_name} does not look like a single self-contained panel "
            f"(found {content.count('<svg')} <svg> tags)"
        )

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_is_non_trivial_size(self, fig7_run, panel_name):
        # Cheap proxy against an accidentally-blank/near-empty export.
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        assert panel_path.stat().st_size > 2000, (
            f"{panel_name} is suspiciously small "
            f"({panel_path.stat().st_size} bytes) for a rendered panel"
        )


class TestPointPlotContent:
    """Frame furniture of the two Alpha-vs-P(Final Color Change) point plots.

    Both keep the "Alpha" x-label and their titles. The Type legend
    (L2L/H2L/L2H/H2H) has moved to its own ``interventions_legend.svg`` panel,
    so it is absent from BOTH point plots. Only ``cell_unit_interventions_
    all_models`` keeps the "P(Final Color Change)" y-label; the rescue panel
    drops it (the two point plots need not share a numeric range). These strings
    appear nowhere else in a matplotlib SVG, so the absence assertions are safe.
    """

    @pytest.mark.parametrize("panel_name", POINT_PLOT_PANELS)
    def test_keeps_x_axis_label(self, fig7_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "Alpha" in content, f"{panel_name} missing x-axis label"

    def test_cell_unit_panel_keeps_y_axis_label(self, fig7_run):
        panel_path = PANELS_DIR / "cell_unit_interventions_all_models.svg"
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "P(Final Color Change)" in content, "missing y-axis label"

    def test_gate_rescue_panel_drops_y_axis_label(self, fig7_run):
        panel_path = PANELS_DIR / "gate_rescue_input_forget.svg"
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "P(Final Color Change)" not in content, (
            "gate_rescue_input_forget should drop its y-axis label "
            "(operator direction 2026-09-04)"
        )

    @pytest.mark.parametrize("panel_name", POINT_PLOT_PANELS)
    def test_type_legend_stripped_from_point_plots(self, fig7_run, panel_name):
        # The Type legend moved to interventions_legend.svg; neither point plot
        # carries it. L2L/H2L/L2H/H2H appear only in that legend text, so their
        # absence here is a reliable check that the legend was stripped.
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        for entry in ("L2L", "H2L", "L2H", "H2H"):
            assert entry not in content, (
                f"{panel_name} still carries Type legend entry {entry!r}; "
                "it should live only in interventions_legend.svg"
            )

    def test_standalone_legend_has_type_entries(self, fig7_run):
        panel_path = PANELS_DIR / "interventions_legend.svg"
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        for entry in ("L2L", "H2L", "L2H", "H2H"):
            assert entry in content, (
                f"interventions_legend.svg missing legend entry {entry!r}"
            )

    def test_cell_unit_panel_title(self, fig7_run):
        panel_path = PANELS_DIR / "cell_unit_interventions_all_models.svg"
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        # Title wraps over two lines in the source
        # ('All Models Cell\nUnit Interventions'); text elements may split
        # per-line, so check both halves rather than the joined string.
        assert "All Models Cell" in content
        assert "Unit Interventions" in content

    def test_gate_rescue_panel_title_uses_unquoted_gate_pair(self, fig7_run):
        panel_path = PANELS_DIR / "gate_rescue_input_forget.svg"
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        # Published title is "All Models (i, f) Gate Rescue Interventions" --
        # unquoted, comma-space gate pair (NOT Python's tuple repr
        # "('i', 'f')"). See the module docstring's "Scope notes".
        assert "(i, f)" in content, (
            "expected the unquoted gate-pair label '(i, f)'; "
            "a literal Python tuple repr (\"('i', 'f')\") would not match"
        )
        assert "Gate" in content
        assert "Rescue Interventions" in content


class TestGateScatterContent:
    """Published text shared by the two Delta-Forget-vs-Delta-Input
    scatters: axis labels (with '(unit-mean)' suffix on the aggregated
    panel only) and the Blue/Green/Red + Unity legend.
    """

    def test_single_model_scatter_axis_labels(self, fig7_run):
        panel_path = PANELS_DIR / "gate_scatter_delta_forget_input.svg"
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "Delta Forget Gate Activity" in content
        assert "Delta Input Gate Activity" in content
        # Should NOT be the unit-mean variant's suffixed labels.
        assert "Delta Forget Gate Activity (unit-mean)" not in content

    def test_aggregated_scatter_axis_labels(self, fig7_run):
        panel_path = PANELS_DIR / "gate_scatter_delta_forget_input_unit_mean.svg"
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "Delta Forget Gate Activity (unit-mean)" in content
        assert "Delta Input Gate Activity (unit-mean)" in content

    def test_aggregated_scatter_title(self, fig7_run):
        panel_path = PANELS_DIR / "gate_scatter_delta_forget_input_unit_mean.svg"
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        # Published title: "Delta (High Hz - Low Hz) Forget vs Input Unit-Mean by
        # Color Entered × Model" (title wraps across two lines in the
        # source; check the pieces rather than the joined/wrapped string).
        assert "Delta (High Hz - Low Hz) Forget vs Input" in content
        assert "Unit-Mean by Color Entered" in content
        assert "Model" in content

    def test_single_model_scatter_has_no_title(self, fig7_run):
        # The source has the title call commented out and the published panel
        # shows no title -- unlike its aggregated sibling, which
        # both title AND axis labels distinguish. This just guards against
        # accidentally reusing the aggregated panel's title text here.
        panel_path = PANELS_DIR / "gate_scatter_delta_forget_input.svg"
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "Unit-Mean by Color Entered" not in content

    @pytest.mark.parametrize("panel_name", GATE_SCATTER_PANELS)
    def test_has_color_entered_and_unity_legend(self, fig7_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        for entry in ("Blue", "Green", "Red", "Unity"):
            assert entry in content, f"{panel_name} missing legend entry {entry!r}"


class TestTransformMemoization:
    def test_cache_dir_populated_after_headless_run(self, fig7_run):
        result, cache_dir = fig7_run
        if result.returncode != 0:
            pytest.fail(f"headless run failed:\n{result.stderr}")
        assert cache_dir.exists(), f"transform cache dir was never created: {cache_dir}"
        cached_files = [p for p in cache_dir.rglob("*") if p.is_file()]
        assert cached_files, (
            f"transform cache dir {cache_dir} has no cached results -- fig7's "
            "gate-intervention frame construction (a render-only read from "
            "data/cache/interventions/) should still be memoized via "
            "transforms.py's shared joblib Memory"
        )


class TestRenderOnly:
    def test_script_does_not_import_hmdcpd(self):
        # hmdcpd-analysis is a separate repo: its code is ported into this
        # one, never imported from it.
        if not FIG7_SCRIPT.exists():
            pytest.skip("fig7_gates.py not found")
        source = FIG7_SCRIPT.read_text()
        assert "import hmdcpd" not in source
        assert "from hmdcpd" not in source

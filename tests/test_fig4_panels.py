"""Integration tests for the NOT-YET-WRITTEN ``figures/fig4_identifying_units.py``
(figures/SPEC.md procedure step 4: port from ``DS2-Identifying-Critical-Units.py``).

Panel inventory (from ``google-drive/paper/png/Fig 4 - Identifying Crit units-1.png``,
the fig4 deck's page 1 -- SPEC scopes each figure to page 1 of its deck)
==========================================================================
Page 1 lays out seven boxes in a 2-column, 4-row grid, lettered A-G top to
bottom, left column then right column:

  A - "Schematic of ElasticNet Reg process": hand-drawn placeholder box (plain
      text on a bordered rectangle, no plotted data). Per SPEC's fig1 treatment
      ("hand-drawn schematic; no script needed") and the fig5-contract
      precedent (schematic boxes A/B on that deck were ruled OUT), this is
      Illustrator compose-time art -- NOT a generated panel. EXCLUDED.
  B - Hazard-rate ElasticNet score curve (binary hz decoder). Line plot, log-x
      "ElasticNet Alpha" (1e0 -> 1e-6) vs. y "Score" (0.0-1.0). Legend: "F1",
      "Accuracy" (single lines -- hz is a binary decode, so no per-label
      split), a horizontal dashed "Chance: 50%" line, and a dash-dot vertical
      "Accuracy: 97%" line marking the last-non-chance alpha. RENDERED -> IN.
  C - Hazard-rate coefficient heatmap. ``pcolor`` over the same log-x alpha
      axis; y-axis is unit index, tick labels "H00".."H14"/"C00".."C14"
      (every other of 32 units: 16 hidden + 16 cell, hidden units plotted
      below cell units) with the vertical axis label "(H)idden / (C)ell Unit
      Number"; colorbar labelled "Coefficient Value"; same dash-dot alpha
      vline as B. RENDERED -> IN.
  D - "Aggregate of identified units / Outer product of hidden and cell unit
      betas to identify pairs": hand-drawn placeholder box, same rationale as
      A. EXCLUDED.
  E - Contingency ElasticNet score curve (3-label decoder, ``cont_r`` in
      DS2 -- see "Judgment calls" below). Same axes as B. Legend: "F1 - Label
      0", "F1 - Label 1", "F1 - Label 2", "Accuracy" (per-label F1 -- 3-class
      decode), horizontal dashed "Chance: 33%" (= 1/3), dash-dot vertical
      "Accuracy: 57%". RENDERED -> IN.
  F - Contingency coefficient heatmap. Same layout/axis conventions as C
      (same 32-unit tick labels, "Coefficient Value" colorbar), different
      coefficient-value color range (deck ticks roughly +-0.15 -- NOT
      asserted here since it is data/fit-dependent). RENDERED -> IN.
  G - "Aggregate of identified units / Outer product of hidden and cell unit
      betas to identify pairs": hand-drawn placeholder box, identical text to
      D. EXCLUDED.

Panel -> stable semantic SVG name (SPEC rule 5), under ``outputs/panels/fig4/``:
  B -> score_curves_hazard_rate.svg
  C -> coef_heatmap_hazard_rate.svg
  E -> score_curves_contingency.svg
  F -> coef_heatmap_contingency.svg

Source mapping (SPEC's fig4 row + ruling 8; ``DS*`` = hmdcpd-analysis/notebooks/):
  * Renderer: ``DS2-Identifying-Critical-Units.py``'s ``plot_coefs_and_metrics``
    (around :607) is DEAD -- it references unbound ``_coefs``/``_hline_chance``
    (should be its own ``coefs``/``hline_chance`` params; a 2-line typo, not a
    logic bug) and would raise ``NameError`` at call time. The CORRECT form is
    ``DS2.1-Identifying-Critical-Units-Grayzone.py``'s copy of the same
    function (:614-682, confirmed by direct diff of both cells): identical
    body with those two lines fixed to reference the bound parameters. SPEC
    ruling 8 directs porting DS2's renderer with this 2-line fix applied,
    matching DS2.1's version -- DS2.1/grayzone itself (the "color" 3-label
    decoder it plots) is out of scope for page 1.
  * Panel B/C data: the hz (hazard-rate) regression-pipeline cell, chance line
    ``hline_chance=0.5`` (DS2 ~:710), binary "accuracy"/"f1" metrics.
  * Panel E/F data: a 3-label contingency decode with ``hline_chance=1/3``.
    DS2 has a ``cont_r`` (regression-cast) pipeline cell nearby (~:753-772)
    that calls the same renderer with ``hline_chance=1/3`` -- this is very
    likely panel E/F's source per SPEC's explicit pointer ("contingency-
    regression cells in DS2"). NOTE (judgment call, flagged for the
    implementer/verifier): as literally read, that DS2 cell passes
    ``metrics_to_plot=['accuracy']`` (reused from the earlier hz cell's
    variable, a marimo-reactive value never reassigned for the cont_r cell),
    which alone would not reproduce panel E's per-label "F1 - Label 0/1/2"
    lines even though ``dict_metrics['f1']`` is computed with ``average=None``
    for ``cont_r`` (its regressor is ``reg_linear``) -- i.e. the data to
    render those lines exists, but this literal cell wouldn't plot it. DS2
    also has a fully-commented-out ``cont_c`` (direct 3-class classification)
    cell using literal ``"L{label} - ..."`` unit-number formatting, which is
    suggestively close to "Label" wording but is dead code, not a runnable
    source. The deck image is authoritative for panel *content*; this test's
    assertions are written against the rendered deck text, not against which
    DS2 cell produced it. The implementer should pass
    ``metrics_to_plot=['accuracy', 'f1']`` to reproduce the deck; exactly
    which upstream decoder (``cont_r`` vs. a corrected ``cont_c``) is a
    judgment call for the implementer/verifier, not settled by this test.

Judgment calls for verifier attention
--------------------------------------
  * Panel titles (e.g. "HZ Metrics with Decreasing ElasticNet Alpha") are
    generated by ``plot_coefs_and_metrics``/``plot_coefficient_analysis`` in
    the ported sources, but no title text is visibly rendered above axes B/C/
    E/F in the deck PNG (top of each panel's plot area starts directly at the
    tick labels). This may mean the deck crop trimmed title rows, or titles
    were disabled for the publication deck. This test does NOT assert any
    title string, to avoid failing for the wrong reason if the implementer's
    titles differ from a guess. It DOES assert axis labels, legend text, and
    tick labels that are unambiguously visible in the deck image. If the
    implementer adds titles, SPEC ruling 6 (fig5's "CT"/"HZ" shorthand via
    ``paper_style.SHORTENED_CONDITIONS``) should apply for consistency, though
    that ruling was stated for fig5 specifically.
  * Deck panel E/F's exact source cell (``cont_r`` vs. a resurrected
    ``cont_c``) is unresolved -- see the source-mapping note above. Whichever
    the implementer picks, this test's content assertions (per-label F1
    lines, "Chance: 33%") should still hold; only the analysis-fidelity
    question (does it match DS2's *originally intended* pipeline) is
    unsettled, and out of scope for a black-box panel-output contract.
  * Intercept panel: ``plot_coefficient_analysis``
    (``critical_units_plots.py``) and (commented-out in both DS sources)
    ``plot_coefs_and_metrics`` both have a 3rd "Intercept Value" subplot; it
    is not rendered on the deck page-1 crop, consistent with both DS sources'
    intercept code being commented out. Not included in EXPECTED_PANELS.
  * fig4's ElasticNet fits run inline in the figure script (SPEC rule 4's
    tolerated exception) -- cold runs are slow. This fixture uses a generous
    subprocess timeout and the same ``LIC_FIG_CACHE_DIR`` isolation seam as
    fig5's contract so the memoization claim (SPEC rule 8) is testable
    without depending on -- or polluting -- the real
    ``data/cache/fig_transforms``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG4_SCRIPT = REPO_ROOT / "figures" / "fig4_identifying_units.py"
PANELS_DIR = REPO_ROOT / "outputs" / "panels" / "fig4"

EXPECTED_PANELS = [
    "score_curves_hazard_rate.svg",   # B
    "coef_heatmap_hazard_rate.svg",   # C
    "score_curves_contingency.svg",   # E
    "coef_heatmap_contingency.svg",   # F
]

# Panels whose ElasticNet decode is binary (single F1/Accuracy line, no
# per-label split) vs. the 3-label contingency decode (per-label F1 lines).
SCORE_CURVE_PANELS = {
    "score_curves_hazard_rate.svg": "hazard_rate",
    "score_curves_contingency.svg": "contingency",
}
HEATMAP_PANELS = [
    "coef_heatmap_hazard_rate.svg",
    "coef_heatmap_contingency.svg",
]

pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def fig4_run(tmp_path_factory):
    """Run figures/fig4_identifying_units.py headlessly, once per test session.

    Overrides the transforms cache-dir seam (``LIC_FIG_CACHE_DIR``, see
    tests/test_fig_transforms.py and tests/test_fig5_panels.py) to an isolated
    tmp dir so this test proves the memoization contract without depending on
    -- or polluting -- the real ``data/cache/fig_transforms``. Panel SVGs, by
    contrast, are written to the REAL ``outputs/panels/fig4/`` -- that is the
    actual deliverable this script exists to produce (SPEC architecture
    table), not a test fixture.

    Timeout is generous (30 min) relative to fig5's 600s: fig4's ElasticNet
    fits run inline in the script (SPEC rule 4's tolerated exception) and a
    cold run sweeps many alphas per decoder; the memoization test below
    exists precisely so a *second* run would be fast, but this fixture only
    ever runs the script once (module-scoped) so it always pays the cold cost.
    """
    cache_dir = tmp_path_factory.mktemp("fig_transforms_cache")
    env = dict(os.environ)
    env["LIC_FIG_CACHE_DIR"] = str(cache_dir)

    result = subprocess.run(
        [sys.executable, str(FIG4_SCRIPT)],
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
        assert FIG4_SCRIPT.exists(), f"not yet written: {FIG4_SCRIPT}"

    def test_exits_zero(self, fig4_run):
        result, _ = fig4_run
        assert result.returncode == 0, (
            "figures/fig4_identifying_units.py exited "
            f"{result.returncode}\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
        # A nonzero exit here also catches SPEC ruling 8's dead-renderer bug
        # (unbound ``_coefs``/``_hline_chance`` -> NameError) if the port
        # forgot the 2-line fix: the dead form crashes rather than silently
        # producing wrong/empty output, so this assertion alone is a
        # sufficient regression guard for that ruling.


class TestPanelOutputs:
    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_writes_expected_panel(self, fig4_run, panel_name):
        result, _ = fig4_run
        if result.returncode != 0:
            pytest.fail(f"headless run failed:\n{result.stderr}")
        panel_path = PANELS_DIR / panel_name
        assert panel_path.exists(), f"expected panel not written: {panel_path}"

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_has_live_text_elements(self, fig4_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "<text" in content, f"{panel_name} has no live <text> elements"

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_is_a_single_self_contained_svg(self, fig4_run, panel_name):
        # Cheap proxy for "panel, not a composed multi-panel grid" (SPEC
        # rule 1) -- exactly one <svg> root element in the document.
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert content.count("<svg") == 1, (
            f"{panel_name} does not look like a single self-contained panel "
            f"(found {content.count('<svg')} <svg> tags)"
        )

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_is_non_trivial_size(self, fig4_run, panel_name):
        # Cheap proxy against an accidentally-blank/near-empty export.
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        assert panel_path.stat().st_size > 2000, (
            f"{panel_name} is suspiciously small "
            f"({panel_path.stat().st_size} bytes) for a rendered panel"
        )


class TestScoreCurveContent:
    """Deck-verified text for panels B (hazard rate) and E (contingency).

    Both share axis labels; they differ in whether F1 is a single line
    ("F1") or split per label ("F1 - Label 0/1/2"), and in their chance
    line's fixed percentage (50% binary vs. 33% = 1/3 for the 3-label
    decode). Neither the exact "Accuracy: NN%" vline value nor any panel
    title is asserted -- see the module docstring's "Judgment calls" section.
    """

    @pytest.mark.parametrize("panel_name", list(SCORE_CURVE_PANELS))
    def test_has_shared_axis_labels(self, fig4_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "ElasticNet Alpha" in content, f"{panel_name} missing x-axis label"
        assert "Score" in content, f"{panel_name} missing y-axis label"
        assert "Accuracy" in content, f"{panel_name} missing an 'Accuracy' legend entry"

    def test_hazard_rate_curve_is_binary(self, fig4_run):
        panel_path = PANELS_DIR / "score_curves_hazard_rate.svg"
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "Chance: 50%" in content, "hz decode chance line should read 'Chance: 50%'"
        assert re.search(r"Accuracy:\s*\d+%", content), (
            "expected an 'Accuracy: NN%' vline legend entry"
        )
        # Binary decode: a bare "F1" legend entry, not a per-label split.
        assert "F1" in content
        assert "F1 - Label" not in content, (
            "hazard-rate decode is binary; should not have per-label F1 lines"
        )

    def test_contingency_curve_is_three_label(self, fig4_run):
        panel_path = PANELS_DIR / "score_curves_contingency.svg"
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "Chance: 33%" in content, "3-label decode chance line should read 'Chance: 33%' (1/3)"
        assert re.search(r"Accuracy:\s*\d+%", content), (
            "expected an 'Accuracy: NN%' vline legend entry"
        )
        for label in ("F1 - Label 0", "F1 - Label 1", "F1 - Label 2"):
            assert label in content, f"contingency decode missing per-label legend entry {label!r}"


class TestCoefficientHeatmapContent:
    """Deck-verified text for panels C (hazard rate) and F (contingency).

    Both heatmaps share the same axis conventions and unit-index tick
    labelling (16 hidden + 16 cell units, alternating labels shown); the
    coefficient-value color range differs per panel (data/fit-dependent) and
    is deliberately NOT asserted here.
    """

    @pytest.mark.parametrize("panel_name", HEATMAP_PANELS)
    def test_has_shared_axis_and_colorbar_labels(self, fig4_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "ElasticNet Alpha" in content, f"{panel_name} missing x-axis label"
        assert "Coefficient Value" in content, f"{panel_name} missing colorbar label"

    @pytest.mark.parametrize("panel_name", HEATMAP_PANELS)
    def test_has_unit_index_tick_labels(self, fig4_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        # Deck shows every-other of 32 units (16 hidden + 16 cell): at least
        # the extremes on each side should appear as literal tick text.
        for tick in ("H00", "H14", "C00", "C14"):
            assert tick in content, f"{panel_name} missing unit tick label {tick!r}"


class TestTransformMemoization:
    def test_cache_dir_populated_after_headless_run(self, fig4_run):
        result, cache_dir = fig4_run
        if result.returncode != 0:
            pytest.fail(f"headless run failed:\n{result.stderr}")
        assert cache_dir.exists(), f"transform cache dir was never created: {cache_dir}"
        cached_files = [p for p in cache_dir.rglob("*") if p.is_file()]
        assert cached_files, (
            f"transform cache dir {cache_dir} has no cached results -- fig4's "
            "inline ElasticNet fits (SPEC rule 4 tolerated exception) should "
            "still be memoized via transforms.py's joblib Memory (SPEC rule 8)"
        )

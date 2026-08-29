"""Integration tests for the NOT-YET-WRITTEN ``figures/fig7_gates.py``
(figures/SPEC.md procedure step 4: port from the DS6.2/DS6.4/DS4 sources
listed in the fig7 mapping-table row, plus ruling 10).

Panel inventory (from ``google-drive/paper/png/Fig 7 - Network Gates-1.png``,
the fig7 deck's page 1 -- SPEC scopes each figure to page 1 of its deck)
==========================================================================
Page 1 lays out two lettered blocks, A (top-left) and B (top-right), plus two
unlettered scatter plots directly below them:

  A - "Schematic for gate 'rescue' experiments": hand-drawn placeholder box
      (plain text on a bordered rectangle, no plotted data). Per SPEC's fig1
      treatment ("hand-drawn schematic; no script needed") and the
      fig4/fig5-contract precedent (schematic boxes ruled OUT on both those
      decks), this is Illustrator compose-time art -- NOT a generated panel.
      EXCLUDED.
  B, left  - "All Models Cell Unit Interventions" point plot. x "Alpha"
      (0.0-1.0), y "P(Final Color Change)" (~0.2-0.5), legend "Type": L2L /
      H2L / L2H / H2H (flare-palette point-and-line series). This is the
      REUSED cell-unit interventions panel (SPEC ruling 10 / fig7 mapping
      row) -- same rendered content as fig6's "All Models Cell Unit
      Interventions" panel (SPEC ruling 9), but exported into fig7's own
      ``outputs/panels/fig7/`` namespace per rule 5 (panel paths are
      per-figure). RENDERED -> IN.
  B, right - "All Models (i, f) Gate\\nRescue Interventions" point plot. Same
      axes/legend shape as B-left (Alpha vs P(Final Color Change), Type:
      L2L/H2L/L2H/H2H) but data comes from gate-frozen ("rescue")
      interventions rather than cell-unit interventions, and the deck's
      title names the specific gate pair rendered: ``(i, f)`` -- the ONLY
      gate pair in scope for page 1 (SPEC mapping row: "page 1 needs only
      the ('i','f') pair"; the source loops over every ``combinations(
      gate_order, 2)`` pair, of which page 1 renders one). RENDERED -> IN.
  Bottom-left  - untitled scatter. x "Delta Forget Gate Activity"
      (~-0.3..0.3), y "Delta Input Gate Activity" (~-0.3..0.3), legend:
      "Blue" / "Green" / "Red" (color_entered categories) + "Unity" (the
      dashed diagonal reference line). One point per (critical unit x
      color_entered) for a single exemplar model -- NOT literally
      per-trial (see "Judgment calls" below re: the SPEC/task wording
      "per-trial gate scatter"). RENDERED -> IN.
  Bottom-right - "Delta (High Hz - Low Hz) Forget vs Input\\n Unit-Mean by
      Color Entered x Model" scatter. x "Delta Forget Gate Activity
      (unit-mean)", y "Delta Input Gate Activity (unit-mean)", same
      Blue/Green/Red/Unity legend, aggregated (mean) across each model's
      critical units within each (color_entered, model) pair. RENDERED -> IN.

Panel -> stable semantic SVG name (SPEC rule 5), under ``outputs/panels/fig7/``:
  B, left      -> cell_unit_interventions_all_models.svg
  B, right     -> gate_rescue_input_forget.svg
  bottom-left  -> gate_scatter_delta_forget_input.svg
  bottom-right -> gate_scatter_delta_forget_input_unit_mean.svg

Source mapping (SPEC's fig7 row, ruling 10, and
``../../figure-to-code-map.md`` Section "Pre-flight corrections
(2026-08-28 scout audit)"; ``DS*`` = hmdcpd-analysis/notebooks/):
  * cell_unit_interventions_all_models.svg <- ``DS4-Interventions.py:1279-1313``
    (the corrected fig7 mapping-row cite; the map's earlier ``:837`` cite is a
    WRONG pointer to the hidden-unit early-draft cell -- ignore it). Title
    ``'All Models Cell\\nUnit Interventions'``, palette
    ``visualization.get_color_palette(['L2L','L2H','H2L','H2H'], (('flare',4),),
    linspace_range=np.array((0.0, 1.1)))``, ``sns.pointplot(df, x='Alpha',
    y='Value', hue='Type', ...)`` where ``df`` is the concatenated
    per-model ``dict_model_pred_dfs_melted_straight_hz_c`` filtered to
    ``Timestep == 24``.
  * gate_rescue_input_forget.svg <- ``DS6.2-Interventions-and-Gates.py:
    814-920`` (either of the two near-duplicate cells at :803-839 / :888-923
    that loop ``for _gate, dict_gate_model_pred_dfs in
    dict_gate_int_model_pred_dfs_melted_straight_hz.items(): ... plt.title(
    f'All Models {{_gate}} Gate\\nRescue Interventions')`` -- page 1 needs
    only the iteration where ``_gate == ('i', 'f')``, built from
    ``process_model_predictions(..., load_gate=('i','f'))`` at :784-793.
    Same point-plot shape/palette as the cell-unit panel; y is still
    "P(Final Color Change)" despite gate freezing being applied at every
    timestep, not just the final one (source computes the full melted df
    then filters ``Timestep == 24``.)
  * gate_scatter_delta_forget_input.svg <- ``DS6.4-Relative-Gate-Activities.py
    :568-590`` (the ``sns.scatterplot(data=plot_df, x='f', y='i',
    hue='color_entered', ...)`` cell). ``plot_df`` (built :507-550) is a
    SINGLE-MODEL table: critical-unit gate deltas (High Hz minus Low Hz
    rescue-intervention activity, signed so the direction always reads
    "toward the target hazard rate") averaged per ``(color_entered,
    unit_idx)`` for one exemplar model. The gate axes are plotted via
    ``dict_gate_names['f'].title()`` / ``dict_gate_names['i'].title()`` ->
    "Forget" / "Input". Title is commented out in the source
    (``# plt.title(...)``) and the deck shows no title above this panel --
    consistent; do not assert one.
  * gate_scatter_delta_forget_input_unit_mean.svg <- reconstructed per SPEC
    ruling 10 from the STALE BACKUP
    ``hmdcpd-analysis/notebooks/marimo/DS6.4-Relative-Gate-Activities.py:
    1150-1195`` (NO current/live DS6.4 has this cell -- current DS6.4 only
    has the single-model scatter above). The backup's own docstring cell
    (:1148-1155) says this is "Same F-vs-I Delta gate scatter as `LJZf`
    [the single-model cell's marimo id], but averaged across the 16
    critical units within each (color_entered, model) pair. Reduces 480
    points -> 30 (3 colors x 10 models)". Its input dataframe ``plot_df_1``
    (the PER-MODEL, per-unit table, i.e. every model's copy of ``plot_df``
    concatenated) is still live in current ``DS6.4:666-696`` -- only the
    final groupby+scatter cell is missing from the live file. The backup
    computes ``plot_df_1_mean = plot_df_1.groupby(['color_entered',
    'model'], as_index=False)[['i','f','g','o']].mean()`` then plots the
    same x='f'/y='i' scatter with an explicit title (unlike the
    single-model cell) and "(unit-mean)" appended to both axis labels.

Judgment calls for verifier attention
--------------------------------------
  * "Per-trial" scatter wording: the work-unit brief (and SPEC's own mapping
    row) call the bottom-left panel a "per-trial gate scatter", but the
    actual source data (``plot_df``, DS6.4:507-550) is per (critical unit x
    color_entered) for ONE exemplar model -- not per behavioral trial (trial
    identity is averaged out by the ``groupby('color_entered')[...].mean()``
    at :537, then melted/merged back to one row per unit). This test's
    panel name and content assertions follow the DATA (unit-level, single
    model) rather than the "per-trial" label, since the deck image is
    authoritative over prose description (SPEC's own framing, echoed in the
    fig4 test's precedent) and the rendered point count in the deck
    (dozens of points, consistent with ~32 units x 3 colors, not hundreds
    of trials) matches the unit-level reading. Flagging for the verifier in
    case "per-trial" was meant literally and points to a different,
    undiscovered source cell.
  * Which exemplar model backs the single-model scatter: DS6.2's
    ``plot_exp = 'san-4604'`` (used a few cells after the rescue-interventions
    df is built, :780) is the only named single-model handle visible near
    DS6.4's ``plot_df`` construction; DS6.4's own ``plot_df`` cell does not
    itself name an exp_id (it operates on whatever single-model
    ``df_ints_all``/``df_data`` are bound to earlier in that notebook's
    session). This test does NOT assert a specific model identity in the
    single-model scatter's content (no model name is drawn into that panel
    in the deck), so the ambiguity does not affect this test's assertions --
    flagging only so the implementer/verifier picks a concrete, cache-backed
    model for ``plot_df`` deliberately rather than by accident.
  * Gate-pair title formatting: DS6.2's literal f-string
    (``f'All Models {_gate} Gate\\nRescue Interventions'`` where ``_gate`` is
    the python tuple ``('i', 'f')``) would render Python's tuple repr, i.e.
    literally ``"All Models ('i', 'f') Gate\\nRescue Interventions"`` (quoted
    elements, comma-space). The deck instead shows unquoted, comma-space
    text: "All Models (i, f) Gate Rescue Interventions". This test asserts
    the DECK's unquoted form (``"(i, f)"``), not the literal source f-string
    output -- per SPEC's "matches its deck panel in content" verification
    step (procedure step 6) and this workspace's standing rule that the deck
    image is authoritative over source-literal reproduction when they
    disagree. The implementer must format the gate-pair label explicitly
    (e.g. ``", ".join(gate)`` or ``f"{gate[0]}, {gate[1]}"``) rather than
    interpolating the raw tuple.
  * ``dict_gate_names`` (DS6.4) is read-only-cited here, not re-derived; this
    test asserts the deck's literal title-cased strings ("Forget", "Input")
    rather than the dict's internal keys/values, which were not directly
    inspected.
  * Fig7's data dependency on gate-frozen rescue intervention caches
    (``data/cache/interventions/lstm/<exp_id>/hz-cell-fi-gates-frozen-
    centroid-interventions-11-all-states-alphas.npz``, confirmed present for
    all 10 LSTM models) and on ``critical_units/dict_units*.pkl`` for the
    unit-mean aggregation is a render-only read per SPEC rule 4; this test
    does not re-verify the cache contents, only that the headless run
    succeeds and the panels it should produce exist with the expected text.
  * SPEC rule 8's "known shared-from-day-one" transform (the per-model
    intervention frames DS6.2 + DS6.4 both build from ``interventions/``) is
    also fig6's dependency; whichever figure's implementer lands first may
    add it to ``transforms.py``. This test's ``TestTransformMemoization``
    check only requires SOME cached artifact land in the isolated
    ``LIC_FIG_CACHE_DIR`` after a run -- it does not require the shared
    transform to already exist, and does not touch ``transforms.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG7_SCRIPT = REPO_ROOT / "figures" / "fig7_gates.py"
PANELS_DIR = REPO_ROOT / "outputs" / "panels" / "fig7"

EXPECTED_PANELS = [
    "cell_unit_interventions_all_models.svg",       # B, left (reused from fig6)
    "gate_rescue_input_forget.svg",                 # B, right ((i, f) pair)
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
    memoization contract (SPEC rule 8) without depending on -- or polluting --
    the real ``data/cache/fig_transforms``. Panel SVGs, by contrast, are
    written to the REAL ``outputs/panels/fig7/`` -- that is the actual
    deliverable this script exists to produce (SPEC architecture table), not
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
        assert FIG7_SCRIPT.exists(), f"not yet written: {FIG7_SCRIPT}"

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
    """Deck-verified text shared by the two Alpha-vs-P(Final Color Change)
    point plots: axis labels and the Type legend (L2L/H2L/L2H/H2H).
    """

    @pytest.mark.parametrize("panel_name", POINT_PLOT_PANELS)
    def test_has_shared_axis_labels(self, fig7_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "Alpha" in content, f"{panel_name} missing x-axis label"
        assert "P(Final Color Change)" in content, (
            f"{panel_name} missing y-axis label"
        )

    @pytest.mark.parametrize("panel_name", POINT_PLOT_PANELS)
    def test_has_type_legend_entries(self, fig7_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        for entry in ("L2L", "H2L", "L2H", "H2H"):
            assert entry in content, f"{panel_name} missing legend entry {entry!r}"

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

    def test_gate_rescue_panel_title_uses_deck_unquoted_pair(self, fig7_run):
        panel_path = PANELS_DIR / "gate_rescue_input_forget.svg"
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        # Deck text is "All Models (i, f) Gate Rescue Interventions" --
        # unquoted, comma-space gate pair (NOT Python's tuple repr
        # "('i', 'f')"). See module docstring's "Judgment calls" section.
        assert "(i, f)" in content, (
            "expected the deck's unquoted gate-pair label '(i, f)'; "
            "a literal Python tuple repr (\"('i', 'f')\") would not match"
        )
        assert "Gate" in content
        assert "Rescue Interventions" in content


class TestGateScatterContent:
    """Deck-verified text shared by the two Delta-Forget-vs-Delta-Input
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
        # Deck: "Delta (High Hz - Low Hz) Forget vs Input Unit-Mean by
        # Color Entered × Model" (title wraps across two lines in the
        # source; check the pieces rather than the joined/wrapped string).
        assert "Delta (High Hz - Low Hz) Forget vs Input" in content
        assert "Unit-Mean by Color Entered" in content
        assert "Model" in content

    def test_single_model_scatter_has_no_title(self, fig7_run):
        # Source has the title call commented out and the deck shows no
        # title above this panel -- unlike its aggregated sibling, which
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
            "gate-intervention frame construction (render-only per SPEC rule "
            "4, from data/cache/interventions/) should still be memoized via "
            "transforms.py's joblib Memory (SPEC rule 8)"
        )


class TestRenderOnly:
    def test_script_does_not_import_hmdcpd(self):
        # SPEC constraints: "hmdcpd-analysis is a separate repo: port code
        # into this repo; do not import hmdcpd from LIC scripts."
        if not FIG7_SCRIPT.exists():
            pytest.skip("fig7_gates.py not yet written")
        source = FIG7_SCRIPT.read_text()
        assert "import hmdcpd" not in source
        assert "from hmdcpd" not in source

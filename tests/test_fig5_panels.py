"""Integration tests for ``figures/fig5_unit_activity.py``.

These load real cached model states from ``data/cache/`` via a real headless
subprocess run, so they are marked ``slow`` + ``integration`` and should be
excluded from a fast default run (``-m "not slow"``).

Panel inventory
===============
Figure 5 renders three full activity blocks, so the inventory is 12 SVGs =
3 blocks x {hidden, cell} x {timecourse, profile}. Panel names are stable
identifiers — the composed figure links to these paths, so an existing output
is never renamed.

  Block 1 — Hazard Rate, color change (panels C/D/E/F, top row-pair):
    activity_timecourse_hazard_rate_hidden.svg   -> C "Hidden Unit Activity",
        Low/High HZ Trials, ordered-change lines (Change 1..8), from the
        ordered-change renderer in ``figures/fig_hazard_rate_activity.py``
        grouped on "Hazard Rate".
    activity_profile_hazard_rate_hidden.svg      -> D "All Hidden Unit Activity
        Profiles", Step Size vs. Activity Decay scatter, state="hidden".
    activity_timecourse_hazard_rate_cell.svg     -> E "Cell Unit Activity"
        (same source as C, cell unit).
    activity_profile_hazard_rate_cell.svg        -> F "All Cell Unit Activity
        Profiles" scatter (same source as D, state="cell").

  Block 2 — Contingency, color change (panels C/D/E/F, lower row-pair):
    activity_timecourse_contingency_hidden.svg   -> C "Hidden Unit Activity",
        Low/High Cont Trials, "Wall Bounce" vline, Color Change 1..6.
    activity_profile_contingency_hidden.svg      -> D scatter (hidden).
    activity_timecourse_contingency_cell.svg     -> E "Cell Unit Activity".
    activity_profile_contingency_cell.svg        -> F scatter (cell).
    Source: the contingency ordered-change frames (grouped on Contingency) in
    the sibling ``figures/fig_contingency_activity.py``.

  Block 3 — Contingency, NO color change (panels G/H/I/J):
    activity_timecourse_contingency_no_change_hidden.svg -> G "Hidden Unit
        Activity", Low/High Cont Trials, No Color Change 1..7.
    activity_profile_contingency_no_change_hidden.svg    -> H scatter (hidden).
    activity_timecourse_contingency_no_change_cell.svg   -> I "Cell Unit Activity".
    activity_profile_contingency_no_change_cell.svg      -> J scatter (cell).
    Source: the no-change contingency frames plus the no-change bounce diffs
    in the same sibling script.

Scope notes:
  * The contingency sibling scripts hold reusable code shared with fig6; that
    says where the code lives, not which figure the panels belong to. All three
    rendered blocks belong to figure 5.
  * G/H/I/J are fully-rendered plots, not placeholders, so they are IN.
  * The top A/B/C/D boxes are hand-drawn schematics (empty boxes of descriptive
    text), composed externally — NOT generated panels.
  * There is deliberately NO ``hazard_rate_no_change`` panel: schematic box B
    ("HZ, no color changes") has no rendered block, and this contract pins what
    is rendered, not what a placeholder implies.
  * The schematics read "IBO as background", but IBO is absent from
    ``extended_dataset`` and the rendered panels do not plot it — it is a
    compose-time layer, not something the panel scripts synthesize.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG5_SCRIPT = REPO_ROOT / "figures" / "fig5_unit_activity.py"
PANELS_DIR = REPO_ROOT / "figures" / "panels" / "fig5"

EXPECTED_PANELS = [
    # Block 1 — Hazard Rate, color change (panels C/D/E/F)
    "activity_timecourse_hazard_rate_hidden.svg",           # C
    "activity_profile_hazard_rate_hidden.svg",              # D
    "activity_timecourse_hazard_rate_cell.svg",             # E
    "activity_profile_hazard_rate_cell.svg",                # F
    # Block 2 — Contingency, color change (panels C/D/E/F, lower row-pair)
    "activity_timecourse_contingency_hidden.svg",           # C
    "activity_profile_contingency_hidden.svg",              # D
    "activity_timecourse_contingency_cell.svg",             # E
    "activity_profile_contingency_cell.svg",                # F
    # Block 3 — Contingency, no color change (panels G/H/I/J)
    "activity_timecourse_contingency_no_change_hidden.svg",  # G
    "activity_profile_contingency_no_change_hidden.svg",     # H
    "activity_timecourse_contingency_no_change_cell.svg",    # I
    "activity_profile_contingency_no_change_cell.svg",       # J
]

pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def fig5_run(tmp_path_factory):
    """Run figures/fig5_unit_activity.py headlessly, once per test session.

    Overrides the transforms cache-dir seam (``LIC_FIG_CACHE_DIR``, see
    tests/test_fig_transforms.py) to an isolated tmp dir so this test proves
    the memoization contract without depending on -- or polluting -- the real
    ``data/cache/fig_transforms``. Panel SVGs, by contrast, are written to
    the REAL ``figures/panels/fig5/`` -- that is the actual deliverable this
    script exists to produce, not a test fixture.
    """
    cache_dir = tmp_path_factory.mktemp("fig_transforms_cache")
    env = dict(os.environ)
    env["LIC_FIG_CACHE_DIR"] = str(cache_dir)

    result = subprocess.run(
        [sys.executable, str(FIG5_SCRIPT)],
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
        assert FIG5_SCRIPT.exists(), f"figure script not found: {FIG5_SCRIPT}"

    def test_exits_zero(self, fig5_run):
        result, _ = fig5_run
        assert result.returncode == 0, (
            "figures/fig5_unit_activity.py exited "
            f"{result.returncode}\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )


class TestPanelOutputs:
    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_writes_expected_panel(self, fig5_run, panel_name):
        result, _ = fig5_run
        if result.returncode != 0:
            pytest.fail(f"headless run failed:\n{result.stderr}")
        panel_path = PANELS_DIR / panel_name
        assert panel_path.exists(), f"expected panel not written: {panel_path}"

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_has_live_text_elements(self, fig5_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_expected_panel")
        content = panel_path.read_text()
        assert "<text" in content, f"{panel_name} has no live <text> elements"

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_is_a_single_self_contained_svg(self, fig5_run, panel_name):
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


class TestTransformMemoization:
    def test_cache_dir_populated_after_headless_run(self, fig5_run):
        result, cache_dir = fig5_run
        if result.returncode != 0:
            pytest.fail(f"headless run failed:\n{result.stderr}")
        assert cache_dir.exists(), f"transform cache dir was never created: {cache_dir}"
        cached_files = [p for p in cache_dir.rglob("*") if p.is_file()]
        assert cached_files, f"transform cache dir {cache_dir} has no cached results"


class TestDatasetPin:
    def test_extended_dataset_pin_preserved(self):
        # fig_hazard_rate_activity.py pins dataset="extended_dataset"
        # (~line 105); the retrofit must not change that (data for the other
        # datasets' "san-4604" exemplar model may not exist / may differ).
        if not FIG5_SCRIPT.exists():
            pytest.skip("fig5_unit_activity.py not found")
        source = FIG5_SCRIPT.read_text()
        assert 'extended_dataset' in source, (
            "expected the 'extended_dataset' pin to survive the retrofit"
        )

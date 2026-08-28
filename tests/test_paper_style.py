"""Contract tests for the (already implemented) paper figure style module.

Covers ``src/learning_in_context/visualization/paper_style.py`` against
``figures/SPEC.md`` rules 2 (live-text SVG export), 3 (final physical size /
size vocabulary), and the module's own ``apply_style`` / ``save_panel``
contract (SPEC procedure step 1). This module is already implemented, so
these tests are expected to PASS.

All tests monkeypatch ``paper_style.PANELS_DIR`` to a ``tmp_path`` so the
real ``outputs/`` tree is never touched.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from learning_in_context.visualization import paper_style


@pytest.fixture(autouse=True)
def _isolated_panels_dir(tmp_path, monkeypatch):
    """Redirect PANELS_DIR to a throwaway directory for every test."""
    isolated = tmp_path / "panels"
    monkeypatch.setattr(paper_style, "PANELS_DIR", isolated)
    yield isolated


@pytest.fixture(autouse=True)
def _reset_rcparams():
    """Restore matplotlib rcParams after each test (apply_style mutates them)."""
    with matplotlib.rc_context():
        yield


class TestApplyStyle:
    """SPEC rule 2: live-text SVG (svg.fonttype='none'); PDF Type 42 if ever added."""

    def test_sets_svg_fonttype_none(self):
        paper_style.apply_style()
        assert plt.rcParams["svg.fonttype"] == "none"

    def test_sets_pdf_fonttype_42(self):
        paper_style.apply_style()
        assert plt.rcParams["pdf.fonttype"] == 42

    def test_disables_top_and_right_spines(self):
        # apply_style's rc block turns off the top/right spines (ticks style).
        paper_style.apply_style()
        assert plt.rcParams["axes.spines.right"] is False
        assert plt.rcParams["axes.spines.top"] is False

    def test_idempotent(self):
        # Calling apply_style() repeatedly (once per figure script run) must
        # not raise or leave rcParams in a broken state.
        paper_style.apply_style()
        paper_style.apply_style()
        assert plt.rcParams["svg.fonttype"] == "none"


class TestSizeVocabulary:
    """SPEC rule 3: final physical size comes from a named size vocabulary."""

    @pytest.mark.parametrize(
        "name",
        ["FULL_WIDTH", "HALF_WIDTH", "THIRD_WIDTH", "PANEL_SQUARE", "PANEL_TUNING"],
    )
    def test_constant_exists(self, name):
        assert hasattr(paper_style, name)

    def test_width_constants_are_positive_numbers(self):
        for name in ("FULL_WIDTH", "HALF_WIDTH", "THIRD_WIDTH"):
            value = getattr(paper_style, name)
            assert isinstance(value, (int, float))
            assert value > 0

    def test_figsize_constants_are_two_tuples_of_positive_numbers(self):
        for name in ("PANEL_SQUARE", "PANEL_TUNING"):
            value = getattr(paper_style, name)
            assert len(value) == 2
            assert all(isinstance(v, (int, float)) and v > 0 for v in value)

    def test_ordering_matches_named_intent(self):
        # A "full" width panel must be wider than a "half" width panel, which
        # in turn must be wider than a "third" width panel.
        assert paper_style.FULL_WIDTH > paper_style.HALF_WIDTH > paper_style.THIRD_WIDTH


class TestSavePanel:
    """save_panel(fig, fig_no, name) writes outputs/panels/fig<N>/<name>.svg."""

    @pytest.fixture(autouse=True)
    def _apply_style_first(self):
        # Realistic pipeline order: every figure script calls apply_style()
        # once (setting svg.fonttype='none') before rendering any panel.
        paper_style.apply_style()

    def _make_labeled_fig(self, label="Mean Unit Activity"):
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0])
        ax.set_ylabel(label)
        return fig

    def test_writes_to_expected_path(self, _isolated_panels_dir):
        fig = self._make_labeled_fig()
        out_path = paper_style.save_panel(fig, 5, "activity_timecourse_hidden")
        expected = _isolated_panels_dir / "fig5" / "activity_timecourse_hidden.svg"
        assert out_path == expected
        assert out_path.exists()

    def test_returns_a_path(self, _isolated_panels_dir):
        fig = self._make_labeled_fig()
        out_path = paper_style.save_panel(fig, 3, "cwc_straight")
        assert isinstance(out_path, Path)

    def test_fig_no_accepts_int_and_str(self, _isolated_panels_dir):
        fig_int = self._make_labeled_fig()
        fig_str = self._make_labeled_fig()
        path_int = paper_style.save_panel(fig_int, 5, "panel_a")
        path_str = paper_style.save_panel(fig_str, "5", "panel_b")
        assert path_int.parent == path_str.parent
        assert path_int.parent.name == "fig5"

    def test_creates_parent_directories(self, _isolated_panels_dir):
        assert not _isolated_panels_dir.exists()
        fig = self._make_labeled_fig()
        paper_style.save_panel(fig, 7, "gate_delta_scatter")
        assert (_isolated_panels_dir / "fig7").is_dir()

    def test_output_is_svg_with_live_text_elements(self, _isolated_panels_dir):
        label = "Mean Hidden Unit Activity"
        fig = self._make_labeled_fig(label=label)
        out_path = paper_style.save_panel(fig, 5, "activity_timecourse_hidden")

        content = out_path.read_text()
        assert out_path.suffix == ".svg"
        # Live (selectable/editable) text elements, not text-as-paths: SPEC rule 2.
        assert "<text" in content
        # The label string itself should appear verbatim as text content,
        # not merely be present as vector path data.
        assert label in content

    def test_does_not_dirty_real_outputs_dir(self):
        # Regression guard: with the autouse fixture active, the module-level
        # PANELS_DIR must point at the isolated tmp_path, never the real repo.
        assert "tmp" in str(paper_style.PANELS_DIR) or not str(
            paper_style.PANELS_DIR
        ).startswith(str(paper_style._REPO_ROOT / "outputs"))

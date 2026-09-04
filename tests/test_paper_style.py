"""Contract tests for the paper figure style module.

Covers ``src/learning_in_context/visualization/paper_style.py``: live-text SVG
export, the final-physical-size vocabulary, and the module's own
``apply_style`` / ``save_panel`` contract.

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
    """Live-text SVG (svg.fonttype='none'); PDF Type 42 if PDF is ever added."""

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
    """Final physical size comes from a named size vocabulary."""

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
    """save_panel(fig, fig_no, name) writes figures/panels/fig<N>/<name>.svg."""

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
        # Live (selectable/editable) text elements, not text-as-paths.
        assert "<text" in content
        # The label string itself should appear verbatim as text content,
        # not merely be present as vector path data.
        assert label in content

    def test_does_not_dirty_real_outputs_dir(self):
        # Regression guard: with the autouse fixture active, the module-level
        # PANELS_DIR must point at the isolated tmp_path, never the real repo.
        assert "tmp" in str(paper_style.PANELS_DIR) or not str(
            paper_style.PANELS_DIR
        ).startswith(str(paper_style._REPO_ROOT / "figures"))


def _make_decor_fig():
    """A drawn axes with a title, both labels, tick labels, and a legend."""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0], label="series")
    ax.set_title("Panel Title")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Value")
    ax.legend()
    return fig, ax


class TestApplyDecor:
    """PanelDecor + apply_decor: the default is a strict no-op; each opted-in
    field acts on exactly one piece of frame furniture."""

    def test_default_is_noop_on_axes_state(self):
        fig, ax = _make_decor_fig()
        paper_style.apply_decor(ax, paper_style.PanelDecor())
        assert ax.get_title() == "Panel Title"
        assert ax.get_xlabel() == "Frame"
        assert ax.get_ylabel() == "Value"
        assert ax.get_legend() is not None
        assert all(t.get_visible() for t in ax.get_xticklabels())
        assert all(t.get_visible() for t in ax.get_yticklabels())

    def test_default_is_byte_identical_export(self, _isolated_panels_dir):
        # The invariant the pipeline promises: a panel that does not opt in
        # re-exports byte-for-byte, so an un-decorated panel and one passed
        # through PanelDecor() produce identical SVGs.
        paper_style.apply_style()
        fig_a, _ = _make_decor_fig()
        path_a = paper_style.save_panel(fig_a, 9, "decor_control")
        bytes_a = path_a.read_bytes()

        fig_b, ax_b = _make_decor_fig()
        paper_style.apply_decor(ax_b, paper_style.PanelDecor())
        path_b = paper_style.save_panel(fig_b, 9, "decor_control")
        assert path_b.read_bytes() == bytes_a

    def test_hides_title_with_none(self):
        fig, ax = _make_decor_fig()
        paper_style.apply_decor(ax, paper_style.PanelDecor(title=None))
        assert ax.get_title() == ""

    def test_overrides_label_with_string(self):
        fig, ax = _make_decor_fig()
        paper_style.apply_decor(ax, paper_style.PanelDecor(xlabel="Alpha"))
        assert ax.get_xlabel() == "Alpha"

    def test_hides_ylabel_with_none(self):
        fig, ax = _make_decor_fig()
        paper_style.apply_decor(ax, paper_style.PanelDecor(ylabel=None))
        assert ax.get_ylabel() == ""
        # ylabel=None hides the label ONLY — the y tick numbers survive. This is
        # what distinguishes it from PanelDecor.shared_y(), and it is the exact
        # behaviour fig7's rescue panel relies on (operator direction
        # 2026-09-04: drop the y-label, keep the y tick numbers).
        assert all(t.get_visible() for t in ax.get_yticklabels())

    def test_blanks_tick_labels_via_tick_params(self):
        fig, ax = _make_decor_fig()
        paper_style.apply_decor(
            ax, paper_style.PanelDecor(xticklabels=False, yticklabels=False)
        )
        assert not any(t.get_visible() for t in ax.get_xticklabels())
        assert not any(t.get_visible() for t in ax.get_yticklabels())

    def test_strips_existing_legend(self):
        fig, ax = _make_decor_fig()
        assert ax.get_legend() is not None
        paper_style.apply_decor(ax, paper_style.PanelDecor(legend=False))
        assert ax.get_legend() is None

    def test_legend_true_is_noop(self):
        # legend=True is an explicit no-op: apply_decor has no handles to draw.
        fig, ax = _make_decor_fig()
        paper_style.apply_decor(ax, paper_style.PanelDecor(legend=True))
        assert ax.get_legend() is not None

    def test_shared_y_constructor(self):
        fig, ax = _make_decor_fig()
        paper_style.apply_decor(ax, paper_style.PanelDecor.shared_y())
        assert ax.get_ylabel() == ""
        assert not any(t.get_visible() for t in ax.get_yticklabels())

    def test_no_legend_constructor(self):
        fig, ax = _make_decor_fig()
        paper_style.apply_decor(ax, paper_style.PanelDecor.no_legend())
        assert ax.get_legend() is None

    def test_convenience_constructor_accepts_overrides(self):
        decor = paper_style.PanelDecor.shared_y(legend=False)
        assert decor.ylabel is None
        assert decor.yticklabels is False
        assert decor.legend is False


class TestMakeLegendPanel:
    """make_legend_panel renders a legend alone on its own figure."""

    @pytest.fixture(autouse=True)
    def _apply_style_first(self):
        paper_style.apply_style()

    def test_only_artist_is_the_legend(self):
        fig, ax = _make_decor_fig()
        handles, labels = ax.get_legend_handles_labels()
        leg_fig = paper_style.make_legend_panel(handles, labels, title="Type")
        # No data axes on the legend figure; its sole artist is the legend.
        assert leg_fig.axes == []
        assert len(leg_fig.legends) == 1

    def test_save_panel_writes_live_text_legend(self, _isolated_panels_dir):
        fig, ax = _make_decor_fig()
        ax.plot([0, 1], [0, 1], label="L2L")
        handles, labels = ax.get_legend_handles_labels()
        leg_fig = paper_style.make_legend_panel(handles, labels, title="Type")
        out_path = paper_style.save_panel(leg_fig, 9, "legend_panel")
        content = out_path.read_text()
        assert out_path.suffix == ".svg"
        assert "<text" in content
        assert "L2L" in content

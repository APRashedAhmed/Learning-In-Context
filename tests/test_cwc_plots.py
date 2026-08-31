"""Contract tests for the CWC swarm+mean renderer.

Covers ``src/learning_in_context/visualization/cwc_plots.py``: a categorical
renderer for confidence-weighted-choice (CWC) panels that draws every raw
observation as a swarm of dots and overlays the per-category mean as a
connected marker-and-line trace.

Two call shapes are pinned:

* a two-level-hue mode (e.g. an ordinal x-category split by a two-level
  condition), which draws a legend unless explicitly suppressed;
* a single-family categorical mode (hue omitted, or equal to ``x``), which
  never draws a legend even though the x-axis itself is a multi-level
  category.

All tests build small synthetic tidy frames — no real data, no disk I/O
except where a test explicitly writes into ``tmp_path``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _reset_rcparams():
    """Restore rcParams and close figures after each test.

    ``apply_style`` mutates rcParams globally, and pyplot retains every figure
    until it is closed.
    """
    with matplotlib.rc_context():
        yield
    plt.close("all")


@pytest.fixture()
def cwc_plots():
    # Imported inside a fixture rather than at module scope so an import
    # failure surfaces as a failing test with a clear ModuleNotFoundError
    # instead of a collection error for the whole file.
    from learning_in_context.visualization import cwc_plots

    return cwc_plots


def _line_y_values(ax):
    """Flatten the y-data of every Line2D on ``ax`` (the mean+line marks)."""
    values = []
    for line in ax.lines:
        values.extend(np.asarray(line.get_ydata(), dtype=float).tolist())
    return [v for v in values if not np.isnan(v)]


def _swarm_y_values(ax):
    """Sorted y-coordinates of every scatter point on ``ax`` (the swarm layer)."""
    values = []
    for collection in ax.collections:
        if not isinstance(collection, mcoll.PathCollection):
            continue
        offsets = np.asarray(collection.get_offsets(), dtype=float)
        if offsets.size:
            values.extend(offsets[:, 1].tolist())
    return sorted(values)


def _luminance(color):
    """Perceptual-ish lightness of a matplotlib color, for ramp-order checks."""
    red, green, blue = mcolors.to_rgb(color)
    return 0.299 * red + 0.587 * green + 0.114 * blue


def _hazard_df():
    # Three x-categories crossed with a two-level hue, three replicate
    # observations per cell. No cell mean equals any of its own replicate
    # values, so a renderer that drew raw observations as lines instead of
    # group means cannot satisfy the mean-correctness tests; no mean is 0.0
    # either, so a stray reference line at zero cannot satisfy them.
    rows = []
    means = {
        (0, "Low"): (-1.0, -0.9, -0.5),
        (0, "High"): (-0.9, -0.5, -0.4),
        (1, "Low"): (-0.8, -0.4, -0.3),
        (1, "High"): (-0.5, -0.2, -0.2),
        (2, "Low"): (-0.4, -0.1, -0.1),
        (2, "High"): (-0.2, 0.2, 0.3),
    }
    for (position, hazard), values in means.items():
        for value in values:
            rows.append(
                {"Grayzone Position": position, "Hazard Rate": hazard, "cwc": value}
            )
    return pd.DataFrame(rows), {k: float(np.mean(v)) for k, v in means.items()}


def _contingency_df():
    # Three x-categories, no independent hue dimension — the family itself
    # is the categorical x-axis (the "Contingency" mode). Cell means again
    # avoid both their own replicate values and 0.0.
    rows = []
    means = {
        "Low": (-1.0, -0.8, -0.3),
        "Medium": (-0.5, -0.3, 0.2),
        "High": (0.1, 0.5, 0.6),
    }
    for level, values in means.items():
        for value in values:
            rows.append({"Contingency": level, "cwc": value})
    return pd.DataFrame(rows), {k: float(np.mean(v)) for k, v in means.items()}


class TestApiShape:
    def test_module_is_importable(self, cwc_plots):
        assert hasattr(cwc_plots, "plot_cwc_swarm")

    def test_returns_figure_and_axes(self, cwc_plots):
        df, _ = _contingency_df()
        result = cwc_plots.plot_cwc_swarm(df, x="Contingency", y="cwc")
        assert isinstance(result, tuple)
        assert len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_accepts_existing_axes(self, cwc_plots):
        df, _ = _contingency_df()
        fig, ax = plt.subplots()
        returned_fig, returned_ax = cwc_plots.plot_cwc_swarm(
            df, x="Contingency", y="cwc", ax=ax
        )
        assert returned_ax is ax
        assert returned_fig is fig

    def test_ylim_applied_exactly(self, cwc_plots):
        df, _ = _contingency_df()
        _, ax = cwc_plots.plot_cwc_swarm(
            df, x="Contingency", y="cwc", ylim=(-1.05, 1.05)
        )
        assert ax.get_ylim() == pytest.approx((-1.05, 1.05))

    def test_ylabel_applied(self, cwc_plots):
        df, _ = _contingency_df()
        _, ax = cwc_plots.plot_cwc_swarm(
            df, x="Contingency", y="cwc", ylabel="Confidence Weighted Choice"
        )
        assert ax.get_ylabel() == "Confidence Weighted Choice"

    def test_xlabel_applied(self, cwc_plots):
        df, _ = _contingency_df()
        _, ax = cwc_plots.plot_cwc_swarm(
            df, x="Contingency", y="cwc", xlabel="Bounce Contingency"
        )
        assert ax.get_xlabel() == "Bounce Contingency"


class TestHazardMode:
    """Two-level hue: swarm + mean per (x-category x hue), legend by default."""

    def test_draws_swarm_and_mean_marks(self, cwc_plots):
        df, _ = _hazard_df()
        _, ax = cwc_plots.plot_cwc_swarm(
            df,
            x="Grayzone Position",
            y="cwc",
            hue="Hazard Rate",
            ylim=(-1.05, 1.05),
        )
        # Every raw observation is drawn once, at its own y-value.
        swarm_y = _swarm_y_values(ax)
        assert len(swarm_y) == len(df)
        assert swarm_y == pytest.approx(sorted(df["cwc"].tolist()))
        # Per-hue mean+line marks are drawn as Line2D artists.
        assert len(ax.lines) >= 2

    def test_legend_present_with_hue_labels_by_default(self, cwc_plots):
        df, _ = _hazard_df()
        _, ax = cwc_plots.plot_cwc_swarm(
            df,
            x="Grayzone Position",
            y="cwc",
            hue="Hazard Rate",
            palette_labels=["Low", "High"],
        )
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = {t.get_text() for t in legend.get_texts()}
        assert {"Low", "High"} <= legend_texts

    def test_legend_suppressed_when_requested(self, cwc_plots):
        df, _ = _hazard_df()
        _, ax = cwc_plots.plot_cwc_swarm(
            df,
            x="Grayzone Position",
            y="cwc",
            hue="Hazard Rate",
            legend=False,
        )
        assert ax.get_legend() is None

    def test_ylim_exact(self, cwc_plots):
        df, _ = _hazard_df()
        _, ax = cwc_plots.plot_cwc_swarm(
            df,
            x="Grayzone Position",
            y="cwc",
            hue="Hazard Rate",
            ylim=(-1.05, 1.05),
        )
        assert ax.get_ylim() == pytest.approx((-1.05, 1.05))

    def test_hue_ramp_runs_light_to_dark(self, cwc_plots):
        df, _ = _hazard_df()
        _, ax = cwc_plots.plot_cwc_swarm(
            df,
            x="Grayzone Position",
            y="cwc",
            hue="Hazard Rate",
            palette="Blues",
            palette_labels=["Low", "High"],
        )
        legend = ax.get_legend()
        assert legend is not None
        handle_colors = {
            text.get_text(): handle.get_color()
            for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
        }
        assert {"Low", "High"} <= set(handle_colors)
        assert _luminance(handle_colors["Low"]) > _luminance(handle_colors["High"])

    def test_ramp_order_and_draw_order_are_independent(self, cwc_plots):
        """``palette_labels`` fixes the ramp; ``hue_order`` fixes the legend order."""
        df, _ = _hazard_df()
        _, ax = cwc_plots.plot_cwc_swarm(
            df,
            x="Grayzone Position",
            y="cwc",
            hue="Hazard Rate",
            palette="Blues",
            palette_labels=["Low", "High"],
            hue_order=["High", "Low"],
        )
        legend = ax.get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == ["High", "Low"]
        handle_colors = {
            text.get_text(): handle.get_color()
            for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
        }
        # The ramp still runs light "Low" -> dark "High" despite the draw order.
        assert _luminance(handle_colors["Low"]) > _luminance(handle_colors["High"])

    def test_participant_count_in_title(self, cwc_plots):
        df, _ = _hazard_df()
        df = df.copy()
        # Six participants, each contributing three rows — so a renderer that
        # reported the row count instead of the participant count is caught.
        df["Participant ID"] = [f"p{i % 6}" for i in range(len(df))]
        assert df["Participant ID"].nunique() == 6
        _, ax = cwc_plots.plot_cwc_swarm(
            df,
            x="Grayzone Position",
            y="cwc",
            hue="Hazard Rate",
            title="Participant CWC",
            participant_count_title=True,
            participant_col="Participant ID",
        )
        title = ax.get_title()
        assert "6" in title
        assert str(len(df)) not in title


class TestContingencyMode:
    """Single categorical family: swarm + mean per x-category, no legend."""

    def test_draws_swarm_and_mean_marks(self, cwc_plots):
        df, _ = _contingency_df()
        _, ax = cwc_plots.plot_cwc_swarm(df, x="Contingency", y="cwc")
        swarm_y = _swarm_y_values(ax)
        assert len(swarm_y) == len(df)
        assert swarm_y == pytest.approx(sorted(df["cwc"].tolist()))
        assert len(ax.lines) >= 1

    def test_no_legend_by_default(self, cwc_plots):
        df, _ = _contingency_df()
        _, ax = cwc_plots.plot_cwc_swarm(df, x="Contingency", y="cwc")
        assert ax.get_legend() is None

    def test_hue_equal_to_x_also_suppresses_legend(self, cwc_plots):
        df, _ = _contingency_df()
        _, ax = cwc_plots.plot_cwc_swarm(
            df, x="Contingency", y="cwc", hue="Contingency"
        )
        assert ax.get_legend() is None

    def test_three_x_categories_present(self, cwc_plots):
        df, _ = _contingency_df()
        _, ax = cwc_plots.plot_cwc_swarm(df, x="Contingency", y="cwc")
        tick_labels = {t.get_text() for t in ax.get_xticklabels()}
        assert {"Low", "Medium", "High"} <= tick_labels

    def test_single_connected_mean_trace(self, cwc_plots):
        """The family mode draws one trace across the categories, not three."""
        df, expected_means = _contingency_df()
        _, ax = cwc_plots.plot_cwc_swarm(df, x="Contingency", y="cwc")
        ordered_means = [expected_means[level] for level in ("Low", "Medium", "High")]
        traces = [
            np.asarray(line.get_ydata(), dtype=float)
            for line in ax.lines
            if len(line.get_ydata()) == len(ordered_means)
        ]
        assert any(
            np.allclose(trace, ordered_means) for trace in traces
        ), f"no connected mean trace matching {ordered_means}"


class TestMeanCorrectness:
    def test_hazard_mean_marks_match_group_means(self, cwc_plots):
        df, expected_means = _hazard_df()
        _, ax = cwc_plots.plot_cwc_swarm(
            df, x="Grayzone Position", y="cwc", hue="Hazard Rate"
        )
        candidates = _line_y_values(ax)
        assert candidates, "expected mean+line marks to be drawn as Line2D artists"
        for mean_value in expected_means.values():
            assert any(
                abs(mean_value - c) < 1e-6 for c in candidates
            ), f"no drawn mean mark near {mean_value}; got {candidates}"

    def test_contingency_mean_marks_match_group_means(self, cwc_plots):
        df, expected_means = _contingency_df()
        _, ax = cwc_plots.plot_cwc_swarm(df, x="Contingency", y="cwc")
        candidates = _line_y_values(ax)
        assert candidates, "expected mean+line marks to be drawn as Line2D artists"
        for mean_value in expected_means.values():
            assert any(
                abs(mean_value - c) < 1e-6 for c in candidates
            ), f"no drawn mean mark near {mean_value}; got {candidates}"


class TestDenseData:
    """Past the swarm's density limit the raw layer still draws every point."""

    def test_family_mode_draws_every_observation(self, cwc_plots):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "Contingency": ["Low", "Medium", "High"] * 100,
                "cwc": rng.uniform(-1.0, 1.0, 300),
            }
        )
        _, ax = cwc_plots.plot_cwc_swarm(df, x="Contingency", y="cwc", swarm_max=10)
        swarm_y = _swarm_y_values(ax)
        assert len(swarm_y) == len(df)
        assert swarm_y == pytest.approx(sorted(df["cwc"].tolist()))

    def test_split_mode_draws_every_observation(self, cwc_plots):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "Grayzone Position": [0, 1, 2] * 100,
                "Hazard Rate": ["Low", "High"] * 150,
                "cwc": rng.uniform(-1.0, 1.0, 300),
            }
        )
        _, ax = cwc_plots.plot_cwc_swarm(
            df, x="Grayzone Position", y="cwc", hue="Hazard Rate", swarm_max=10
        )
        swarm_y = _swarm_y_values(ax)
        assert len(swarm_y) == len(df)
        assert swarm_y == pytest.approx(sorted(df["cwc"].tolist()))


class TestNoSideEffects:
    def test_does_not_touch_disk(self, cwc_plots, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        df, _ = _hazard_df()
        cwc_plots.plot_cwc_swarm(df, x="Grayzone Position", y="cwc", hue="Hazard Rate")
        written = list(tmp_path.iterdir())
        assert written == [], f"rendering wrote unexpected files: {written}"

    def test_does_not_mutate_input_frame(self, cwc_plots):
        df, _ = _hazard_df()
        before = df.copy(deep=True)
        cwc_plots.plot_cwc_swarm(df, x="Grayzone Position", y="cwc", hue="Hazard Rate")
        pd.testing.assert_frame_equal(df, before)

    def test_does_not_apply_a_style_of_its_own(self, cwc_plots):
        """Styling is the caller's: rendering must not touch global rcParams."""
        plt.rcParams["font.size"] = 17.0
        plt.rcParams["axes.spines.top"] = True
        df, _ = _hazard_df()
        cwc_plots.plot_cwc_swarm(df, x="Grayzone Position", y="cwc", hue="Hazard Rate")
        assert plt.rcParams["font.size"] == 17.0
        assert plt.rcParams["axes.spines.top"] is True


class TestLiveText:
    def test_composes_with_apply_style_for_live_svg_text(self, cwc_plots, tmp_path):
        from learning_in_context.visualization import paper_style

        paper_style.apply_style()
        assert plt.rcParams["svg.fonttype"] == "none"

        df, _ = _contingency_df()
        fig, ax = cwc_plots.plot_cwc_swarm(
            df, x="Contingency", y="cwc", ylabel="Confidence Weighted Choice"
        )
        out_path = tmp_path / "cwc_panel.svg"
        fig.savefig(out_path, format="svg")

        content = out_path.read_text()
        assert "<text" in content
        assert "Confidence Weighted Choice" in content

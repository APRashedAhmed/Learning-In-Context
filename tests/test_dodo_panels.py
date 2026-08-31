"""Structural tests for the ``task_panels`` doit task.

Checks the shape of the task graph — sub-task names and declared targets —
via ``doit list``/``doit info``, without ever executing the figure-rendering
pipeline (that is covered by the per-figure integration tests, e.g.
``tests/test_fig4_panels.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import run_doit

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SUBTASKS = ["fig2", "fig3", "fig4", "fig5", "fig6", "fig7"]

EXPECTED_TARGETS = {
    "fig2": [
        "estimate_curve_hazard_rate.svg",
        "estimate_curve_contingency.svg",
        "cwc_hazard_rate.svg",
        "cwc_contingency.svg",
    ],
    "fig3": [
        "cwc_straight_participants.svg",
        "cwc_straight_rnn.svg",
        "cwc_straight_lstm.svg",
        "cwc_bounce_participants.svg",
        "cwc_bounce_rnn.svg",
        "cwc_bounce_lstm.svg",
    ],
    "fig4": [
        "score_curves_hazard_rate.svg",
        "coef_heatmap_hazard_rate.svg",
        "score_curves_contingency.svg",
        "coef_heatmap_contingency.svg",
    ],
    "fig5": [
        "activity_timecourse_hazard_rate_hidden.svg",
        "activity_profile_hazard_rate_hidden.svg",
        "activity_timecourse_hazard_rate_cell.svg",
        "activity_profile_hazard_rate_cell.svg",
        "activity_timecourse_contingency_hidden.svg",
        "activity_profile_contingency_hidden.svg",
        "activity_timecourse_contingency_cell.svg",
        "activity_profile_contingency_cell.svg",
        "activity_timecourse_contingency_no_change_hidden.svg",
        "activity_profile_contingency_no_change_hidden.svg",
        "activity_timecourse_contingency_no_change_cell.svg",
        "activity_profile_contingency_no_change_cell.svg",
    ],
    "fig6": [
        "intervention_timecourse_hz_hidden.svg",
        "summary_pointplot_hz_hidden.svg",
        "intervention_timecourse_hz_cell.svg",
        "summary_pointplot_hz_cell.svg",
        "intervention_timecourse_ct_hidden.svg",
        "summary_pointplot_ct_hidden.svg",
        "intervention_timecourse_ct_cell.svg",
        "summary_pointplot_ct_cell.svg",
    ],
    "fig7": [
        "cell_unit_interventions_all_models.svg",
        "gate_rescue_input_forget.svg",
        "gate_scatter_delta_forget_input.svg",
        "gate_scatter_delta_forget_input_unit_mean.svg",
    ],
}


@pytest.mark.doit
class TestPanelsTaskGraph:
    """Graph-shape checks only — never runs the figure scripts."""

    def test_panels_group_and_subtasks_listed(self):
        result = run_doit("list", "--all")
        assert result.returncode == 0, result.stderr
        listed = result.stdout

        assert "panels " in listed or "panels\t" in listed or "\npanels " in listed
        for fig_name in EXPECTED_SUBTASKS:
            assert f"panels:{fig_name}" in listed, (
                f"doit list --all is missing sub-task panels:{fig_name}:\n{listed}"
            )

    @staticmethod
    def _info(fig_name: str) -> str:
        """The ``doit info`` report for one sub-task.

        ``doit info`` exits non-zero whenever the task it describes is merely
        out of date, which every panel sub-task is until its script has been
        run — so the exit code says nothing about whether the task graph is
        well-formed, and asserting on it would make these graph-shape checks
        pass or fail on whether someone happened to run ``doit panels`` first.
        The emitted report is what carries the answer, and doit emits it in
        either state.
        """
        result = run_doit("info", f"panels:{fig_name}")
        assert f"panels:{fig_name}" in result.stdout, (
            f"doit info panels:{fig_name} reported no such task\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
        return result.stdout

    @pytest.mark.parametrize("fig_name", EXPECTED_SUBTASKS)
    def test_subtask_targets_match_expected_svgs(self, fig_name):
        info = self._info(fig_name)

        out_dir = REPO_ROOT / "figures" / "panels" / fig_name
        for panel_name in EXPECTED_TARGETS[fig_name]:
            expected_target = str(out_dir / panel_name)
            assert expected_target in info, (
                f"panels:{fig_name} is missing target {expected_target}:\n{info}"
            )

    @pytest.mark.parametrize("fig_name", EXPECTED_SUBTASKS)
    def test_subtask_depends_on_shared_style_and_transforms(self, fig_name):
        info = self._info(fig_name)

        assert "visualization/paper_style.py" in info
        assert "visualization/transforms.py" in info

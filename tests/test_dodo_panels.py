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

EXPECTED_SUBTASKS = ["fig4", "fig5", "fig6", "fig7"]

EXPECTED_TARGETS = {
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

    @pytest.mark.parametrize("fig_name", EXPECTED_SUBTASKS)
    def test_subtask_targets_match_expected_svgs(self, fig_name):
        result = run_doit("info", f"panels:{fig_name}")
        assert result.returncode == 0, result.stderr
        info = result.stdout

        out_dir = REPO_ROOT / "figures" / "panels" / fig_name
        for panel_name in EXPECTED_TARGETS[fig_name]:
            expected_target = str(out_dir / panel_name)
            assert expected_target in info, (
                f"panels:{fig_name} is missing target {expected_target}:\n{info}"
            )

    @pytest.mark.parametrize("fig_name", EXPECTED_SUBTASKS)
    def test_subtask_depends_on_shared_style_and_transforms(self, fig_name):
        result = run_doit("info", f"panels:{fig_name}")
        assert result.returncode == 0, result.stderr
        info = result.stdout

        assert "visualization/paper_style.py" in info
        assert "visualization/transforms.py" in info

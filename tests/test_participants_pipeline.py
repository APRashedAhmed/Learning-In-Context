"""Contract tests for the ported participant-response pipeline.

Target module: ``learning_in_context.analysis.participants`` — the jsPsych
response loader, the initial-stats computation, the six-stage participant
filter chain, and the confidence-weighted-choice (CWC) grouping used by the
task-results figures. Target artifact: a pipeline entry point that writes
``data/cache/participants/{participant_stats_filtered.parquet,
participant_cwc.parquet, participant_counts.json}``.

Two behaviors are pinned here as corrected-and-binding, independent of what
an earlier notebook implementation happened to do:

1. Per-participant hazard-rate CWC grouping. The trial metadata used to
   define the (Hazard Rate, Grayzone Position) buckets for a participant's
   Straight-trial CWC is the full experiment metadata table, not anything
   derived from another participant's response set. Concretely:
   ``participant_cwc_by_hazard`` for participant P is invariant to which
   other participants are present in its input — adding or removing another
   participant's data must not change P's own grouped rows.

2. The fast-catch-trial reclassification. Catch trials where the ball's
   final color becomes visible very early (within a small number of
   timesteps of the trial's end) are relabeled from trial type "Catch" to
   "Catch-Fast" so they are excluded from catch-trial accuracy. This
   relabeling must be a real, visible mutation of the trial-type column (via
   label-based assignment on the owning frame), not a chained assignment
   that silently no-ops under copy-on-write semantics. Because catch-trial
   accuracy feeds the final "Accuracy Catch >= 0.8" filter stage, whether
   this relabeling actually takes effect measurably changes which
   participants survive that stage.

House convention: import failures are contained to a single top-level
try/except so this file still collects cleanly and reports one clear failure
per test, rather than aborting collection for the whole test session.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import linregress

try:
    from learning_in_context.analysis import participants
    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - only when the module is absent/broken
    participants = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RAW_DATASET_DIR = _REPO_ROOT / "data" / "raw" / "hbb_v3_2_2"
_RAW_PARTICIPANT_DIR = _RAW_DATASET_DIR / "hbb_participant_responses_v3_2_2"
_DEMOGRAPHICS_CSV = _RAW_DATASET_DIR / "hbb_demographics_v3_2_2.csv"
_PARTICIPANT_DATASET_DIR = (
    _REPO_ROOT / "data" / "cache" / "model_states" / "participant_dataset"
)

_VIDEO_ID = "Video ID"
_PARTICIPANT_ID = "Participant ID"


def test_module_is_importable():
    """Standalone check naming the actual root cause of every other failure."""
    if _IMPORT_ERROR is not None:
        pytest.fail(
            "learning_in_context.analysis.participants failed to import: "
            f"{_IMPORT_ERROR!r}"
        )


# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------

def _trial_metadata(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal trial-metadata frame indexed by Video ID."""
    df = pd.DataFrame(rows)
    df = df.set_index(_VIDEO_ID)
    df.index.name = _VIDEO_ID
    return df


def _straight_metadata() -> pd.DataFrame:
    return _trial_metadata([
        {_VIDEO_ID: "S1", "trial": "Straight", "Hazard Rate": "Low", "idx_time": 0},
        {_VIDEO_ID: "S2", "trial": "Straight", "Hazard Rate": "Low", "idx_time": 1},
        {_VIDEO_ID: "S3", "trial": "Straight", "Hazard Rate": "High", "idx_time": 0},
        {_VIDEO_ID: "S4", "trial": "Straight", "Hazard Rate": "High", "idx_time": 1},
    ])


def _contingency_metadata() -> pd.DataFrame:
    return _trial_metadata([
        {_VIDEO_ID: "B1", "trial": "Bounce", "Contingency": "Low"},
        {_VIDEO_ID: "B2", "trial": "Bounce", "Contingency": "Medium"},
        {_VIDEO_ID: "B3", "trial": "Bounce", "Contingency": "High"},
    ])


def _participant_cwc_frame(rows: dict[str, float]) -> pd.DataFrame:
    """A per-participant response frame indexed by Video ID with a CWC column."""
    df = pd.DataFrame({"CWC": pd.Series(rows)})
    df.index.name = _VIDEO_ID
    return df


# ---------------------------------------------------------------------------
# filter_video_indexed_data — ported verbatim from the shared utility
# ---------------------------------------------------------------------------

class TestFilterVideoIndexedData:
    def test_requires_video_id_index_on_data(self):
        df_data = pd.DataFrame({"x": [1, 2]}, index=pd.Index(["a", "b"], name="wrong"))
        df_filter = pd.DataFrame({"y": [1]}, index=pd.Index(["a"], name=_VIDEO_ID))
        with pytest.raises(ValueError):
            participants.filter_video_indexed_data(df_data, df_filter)

    def test_requires_video_id_index_on_filter(self):
        df_data = pd.DataFrame({"x": [1]}, index=pd.Index(["a"], name=_VIDEO_ID))
        df_filter = pd.DataFrame({"y": [1]}, index=pd.Index(["a"], name="wrong"))
        with pytest.raises(ValueError):
            participants.filter_video_indexed_data(df_data, df_filter)

    def test_one_way_returns_intersection_of_data_rows(self):
        df_data = pd.DataFrame(
            {"x": [1, 2, 3]}, index=pd.Index(["a", "b", "c"], name=_VIDEO_ID)
        )
        df_filter = pd.DataFrame({"y": [1, 1]}, index=pd.Index(["b", "c"], name=_VIDEO_ID))
        result = participants.filter_video_indexed_data(df_data, df_filter)
        assert list(result.index) == ["b", "c"]
        assert list(result["x"]) == [2, 3]

    def test_two_way_returns_both_frames_on_same_intersection(self):
        df_data = pd.DataFrame(
            {"x": [1, 2, 3]}, index=pd.Index(["a", "b", "c"], name=_VIDEO_ID)
        )
        df_filter = pd.DataFrame(
            {"y": [10, 20, 30]}, index=pd.Index(["b", "c", "d"], name=_VIDEO_ID)
        )
        result_data, result_filter = participants.filter_video_indexed_data(
            df_data, df_filter, two_way=True
        )
        assert set(result_data.index) == {"b", "c"}
        assert set(result_filter.index) == {"b", "c"}
        assert list(result_data.index) == list(result_filter.index)


# ---------------------------------------------------------------------------
# Response/confidence utilities
# ---------------------------------------------------------------------------

class TestResponseAccuracy:
    def test_accuracy_vector_compares_response_to_correct_response(self):
        df = pd.DataFrame({"response": [1, 2, 3]})
        df_comparison = pd.DataFrame({"correct_response": [1, 2, 2]})
        vector = participants.response_accuracy_vector(
            df, df_comparison=df_comparison
        )
        assert list(vector) == [True, True, False]

    def test_accuracy_is_mean_of_vector(self):
        df = pd.DataFrame({"response": [1, 2, 3], "correct_response": [1, 2, 2]})
        assert participants.response_accuracy(df) == pytest.approx(2 / 3)

    def test_accuracy_vector_requires_response_column(self):
        df = pd.DataFrame({"not_response": [1]})
        with pytest.raises(ValueError):
            participants.response_accuracy_vector(df)


class TestComputeConfidence:
    def test_average_mode_scales_to_unit_interval(self):
        df = pd.DataFrame({"slider_end": [50.0, 100.0], "slider_start": [0.0, 0.0]})
        assert participants.compute_confidence(df, average=True) == pytest.approx(0.75)

    def test_var_mode_matches_manual_variance(self):
        values = np.array([10.0, 50.0, 90.0])
        df = pd.DataFrame({"slider_end": values, "slider_start": [0.0, 0.0, 0.0]})
        expected = (values / 100).var()
        assert participants.compute_confidence(df, var=True) == pytest.approx(expected)

    def test_mutually_exclusive_reductions_raise(self):
        df = pd.DataFrame({"slider_end": [1.0], "slider_start": [0.0]})
        with pytest.raises(ValueError):
            participants.compute_confidence(df, average=True, median=True)


class TestNodeIdHelpers:
    def test_infer_video_internal_node_id_replaces_final_segment_prefix(self):
        assert (
            participants.infer_video_internal_node_id_from_response("0.0-1.0-2.5")
            == "0.0-1.0-1.5"
        )

    def test_infer_block_from_internal_node_id_reads_third_segment(self):
        assert participants.infer_block_from_internal_node_id("0.0-1.0-3.7") == 3

    def test_ms_to_min_rounds_to_nearest_minute(self):
        assert participants.ms_to_min(np.array([61_000, 89_000])).tolist() == [1, 1]
        assert int(participants.ms_to_min(np.array([90_000]))[0]) == 2


# ---------------------------------------------------------------------------
# Correction 2 — fast-catch-trial reclassification actually applies
# ---------------------------------------------------------------------------

class TestReclassifyFastCatchTrials:
    def _catch_metadata(self) -> pd.DataFrame:
        return _trial_metadata([
            {_VIDEO_ID: "C1", "trial": "Catch"},
            {_VIDEO_ID: "C2", "trial": "Catch"},
            {_VIDEO_ID: "S1", "trial": "Straight"},
        ])

    def test_fast_catch_trials_are_relabeled(self):
        df = self._catch_metadata()
        timesteps = pd.Series({"C1": 3, "C2": 15}, name="timesteps_since_color_stable")
        result = participants.reclassify_fast_catch_trials(
            df, timesteps, min_timesteps=8
        )
        assert result.loc["C1", "trial"] == "Catch-Fast"
        assert result.loc["C2", "trial"] == "Catch"
        assert result.loc["S1", "trial"] == "Straight"

    def test_boundary_timestep_is_reclassified(self):
        # min_timesteps is an inclusive upper bound on "fast": a settle time
        # exactly at the threshold still counts as fast.
        df = self._catch_metadata()
        timesteps = pd.Series({"C1": 8, "C2": 9})
        result = participants.reclassify_fast_catch_trials(
            df, timesteps, min_timesteps=8
        )
        assert result.loc["C1", "trial"] == "Catch-Fast"
        assert result.loc["C2", "trial"] == "Catch"

    def test_does_not_mutate_input_frame(self):
        df = self._catch_metadata()
        original_trial_column = df["trial"].copy()
        timesteps = pd.Series({"C1": 1, "C2": 20})
        participants.reclassify_fast_catch_trials(df, timesteps, min_timesteps=8)
        pd.testing.assert_series_equal(df["trial"], original_trial_column)

    def test_requires_trial_column(self):
        df = pd.DataFrame({"x": [1]}, index=pd.Index(["C1"], name=_VIDEO_ID))
        timesteps = pd.Series({"C1": 1})
        with pytest.raises(ValueError):
            participants.reclassify_fast_catch_trials(df, timesteps, min_timesteps=8)


class TestComputeCatchTimesteps:
    """Pins the diagnostic that feeds ``reclassify_fast_catch_trials``: the
    number of trailing timesteps, before a catch trial's final frame, during
    which the ball's color already matched its final color.
    """

    def _write_samples_csv(self, dir_path: Path, colors: list[tuple[int, int, int]]):
        dir_path.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(colors, columns=["r", "g", "b"])
        df.to_csv(dir_path / "clip_samples.csv", index=False)

    def test_counts_timesteps_since_last_color_difference(self, tmp_path):
        # Fast: color settles to its final value 2 steps before the end.
        fast_dir = tmp_path / "fast"
        self._write_samples_csv(
            fast_dir,
            [(0, 0, 0), (0, 0, 0), (1, 1, 1), (1, 1, 1), (1, 1, 1)],
        )
        # Slow: color keeps changing until the very last step.
        slow_dir = tmp_path / "slow"
        self._write_samples_csv(
            slow_dir,
            [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (3, 3, 3)],
        )
        df_catch = _trial_metadata([
            {_VIDEO_ID: "C1", "trial": "Catch", "Dir Video": str(fast_dir)},
            {_VIDEO_ID: "C2", "trial": "Catch", "Dir Video": str(slow_dir)},
        ])
        result = participants.compute_catch_timesteps(df_catch)
        # C1: rows differing from final (1,1,1) are indices 0,1 -> last such
        # index is 1; last row index is 4 -> 4 - 1 = 3 timesteps since stable.
        assert result["C1"] == 3
        # C2: only the second-to-last row (index 3) matches the final color;
        # the last differing index is 2 -> 4 - 2 = 2.
        assert result["C2"] == 2


# ---------------------------------------------------------------------------
# jsPsych response loader — schema contract on a synthetic participant
# ---------------------------------------------------------------------------

class TestLoadParticipantResponses:
    @pytest.fixture
    def synthetic_participant_dir(self, tmp_path):
        """One participant, two block trials (Straight + Catch), each a
        response/confidence pair plus its stimulus-bearing trial, matching
        the jsPsych trial-record shape used by the real export.
        """
        participant_id = "synthetic-participant"
        participant_dir = tmp_path / "responses" / participant_id
        participant_dir.mkdir(parents=True)

        def video_path(block, video, color):
            return [f"media/videos/block_{block}/video_{video}/clip_{color}.mp4"]

        trials = [
            # -- Straight trial: stimulus, response, confidence --
            {
                "task": "stimulus", "trial_index": 0,
                "internal_node_id": "0.0-1.0-1.5",
                "stimulus": video_path(1, 3, "red"),
                "response": None, "correct_response": None, "slider_start": None,
                "rt": 100, "time_elapsed": 100,
                "participant_id": participant_id, "study_id": "study-1",
            },
            {
                "task": "response", "trial_index": 1,
                "internal_node_id": "0.0-1.0-2.5",
                "stimulus": None,
                "response": 0, "correct_response": 0, "slider_start": None,
                "rt": 200, "time_elapsed": 300,
                "participant_id": participant_id, "study_id": "study-1",
            },
            {
                "task": "confidence", "trial_index": 2,
                "internal_node_id": "0.0-1.0-3.0",
                "stimulus": None,
                "response": 72.0, "correct_response": None, "slider_start": 20.0,
                "rt": 50, "time_elapsed": 350,
                "participant_id": participant_id, "study_id": "study-1",
            },
            # -- Catch trial: stimulus, response, confidence --
            {
                "task": "stimulus", "trial_index": 3,
                "internal_node_id": "0.0-1.0-1.9",
                "stimulus": video_path(1, 9, "green"),
                "response": None, "correct_response": None, "slider_start": None,
                "rt": 100, "time_elapsed": 450,
                "participant_id": participant_id, "study_id": "study-1",
            },
            {
                "task": "response", "trial_index": 4,
                "internal_node_id": "0.0-1.0-4.9",
                "stimulus": None,
                "response": 2, "correct_response": 1, "slider_start": None,
                "rt": 200, "time_elapsed": 650,
                "participant_id": participant_id, "study_id": "study-1",
            },
            {
                "task": "confidence", "trial_index": 5,
                "internal_node_id": "0.0-1.0-5.0",
                "stimulus": None,
                "response": 40.0, "correct_response": None, "slider_start": 10.0,
                "rt": 50, "time_elapsed": 700,
                "participant_id": participant_id, "study_id": "study-1",
            },
        ]
        payload = {"app_platform": "Win32", "trials": trials}
        (participant_dir / "export.json").write_text(json.dumps(payload))
        return tmp_path / "responses", participant_id

    @pytest.fixture
    def synthetic_dataset_metadata(self):
        return _trial_metadata([
            {
                _VIDEO_ID: "S1", "trial": "Straight",
                "Dataset Block": 1, "Dataset Block Video": 3,
            },
            {
                _VIDEO_ID: "C1", "trial": "Catch",
                "Dataset Block": 1, "Dataset Block Video": 9,
            },
        ])

    def test_returns_expected_top_level_keys(
        self, synthetic_participant_dir, synthetic_dataset_metadata
    ):
        responses_dir, participant_id = synthetic_participant_dir
        result = participants.load_participant_responses(
            participant_id,
            responses_dir,
            synthetic_dataset_metadata,
            dir_videos=Path("/does/not/need/to/exist"),
        )
        assert set(result) >= {"device", "all_trials", "responses"}
        assert result["device"] == "Win32"

    def test_splits_responses_by_trial_type_and_video_id(
        self, synthetic_participant_dir, synthetic_dataset_metadata
    ):
        responses_dir, participant_id = synthetic_participant_dir
        result = participants.load_participant_responses(
            participant_id,
            responses_dir,
            synthetic_dataset_metadata,
            dir_videos=Path("/does/not/need/to/exist"),
        )
        responses = result["responses"]
        assert set(responses) >= {"straight", "catch", "experiment", "missed"}

        straight = responses["straight"]
        assert list(straight.index) == ["S1"]
        assert straight.index.name == _VIDEO_ID
        for column in ("response", "correct_response", "rt", "slider_start", "slider_end"):
            assert column in straight.columns

        catch = responses["catch"]
        assert list(catch.index) == ["C1"]

        # "experiment" excludes catch trials.
        experiment = responses["experiment"]
        assert "C1" not in experiment.index
        assert "S1" in experiment.index

    def test_response_and_confidence_values_are_paired_correctly(
        self, synthetic_participant_dir, synthetic_dataset_metadata
    ):
        responses_dir, participant_id = synthetic_participant_dir
        result = participants.load_participant_responses(
            participant_id,
            responses_dir,
            synthetic_dataset_metadata,
            dir_videos=Path("/does/not/need/to/exist"),
        )
        straight = result["responses"]["straight"]
        # response/correct_response are stored 1-indexed (raw jsPsych values
        # are 0-indexed color choices; +1 aligns them with the 1/2/3 color
        # coding used elsewhere in the pipeline).
        assert int(straight.loc["S1", "response"]) == 1
        assert int(straight.loc["S1", "correct_response"]) == 1
        # slider_start/slider_end are pulled from the following confidence
        # trial, not the response trial itself.
        assert straight.loc["S1", "slider_start"] == pytest.approx(20.0)
        assert straight.loc["S1", "slider_end"] == pytest.approx(72.0)


# ---------------------------------------------------------------------------
# Filter-chain stages — pure predicates on crafted stats fixtures
# ---------------------------------------------------------------------------

def _stats_frame(rows: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = _PARTICIPANT_ID
    return df


class TestFilterByStatus:
    def test_excludes_returned_status_participants(self):
        data_all = {"p1": {"marker": 1}, "p2": {"marker": 2}, "p3": {"marker": 3}}
        demographics = pd.DataFrame(
            {"Status": ["APPROVED", "RETURNED", "AWAITING REVIEW"]},
            index=pd.Index(["p1", "p2", "p3"], name=_PARTICIPANT_ID),
        )
        filtered_data, filtered_demographics = participants.filter_by_status(
            data_all, demographics
        )
        assert set(filtered_data) == {"p1", "p3"}
        assert set(filtered_demographics.index) == {"p1", "p3"}


class TestFilterByCompleteness:
    def test_keeps_participants_above_fraction_of_total_trials(self):
        df = _stats_frame({
            "p1": {"Experiment Trials Present": 168},
            "p2": {"Experiment Trials Present": 140},
            "p3": {"Experiment Trials Present": 50},
        })
        result = participants.filter_by_completeness(
            df, total_trials=168, min_fraction=0.8
        )
        # cutoff is 168 * 0.8 = 134.4 -> p2 (140) survives, p3 (50) does not.
        assert set(result.index) == {"p1", "p2"}


class TestFilterByMissed:
    def test_drops_participants_over_missed_fraction(self):
        df = _stats_frame({
            "p1": {"Num Missed": 1},
            "p2": {"Num Missed": 40},
        })
        result = participants.filter_by_missed(df, total_trials=168, max_fraction=0.2)
        # cutoff is 168 * 0.2 = 33.6 -> p2 (40) is dropped.
        assert set(result.index) == {"p1"}


class TestFilterBySliderUse:
    def test_keeps_only_participants_meeting_both_conditions(self):
        """Reporting confidence means both moving the slider off its starting
        position and moving it by different amounts on different trials, so a
        participant has to clear both cutoffs to be kept.
        """
        df = _stats_frame({
            # Moved the slider, and by varying amounts -> kept.
            "p1": {"RMS Slider Change": 0.5, "Var RMS Slider Change": 0.10},
            # Moved the slider the same distance every trial -> dropped.
            "p2": {"RMS Slider Change": 0.5, "Var RMS Slider Change": 0.01},
            # Barely moved the slider -> dropped.
            "p3": {"RMS Slider Change": 0.05, "Var RMS Slider Change": 0.10},
            # Neither -> dropped.
            "p4": {"RMS Slider Change": 0.05, "Var RMS Slider Change": 0.01},
        })
        result = participants.filter_by_slider_use(
            df, rms_cutoff=0.2, var_cutoff=0.03
        )
        assert set(result.index) == {"p1"}


class TestFilterBySliderValues:
    def test_requires_nonzero_variance_and_interior_rms_slider_end(self):
        df = _stats_frame({
            "p1": {"Var Confidence Experiment": 0.02, "RMS Slider End": 0.5},   # kept
            "p2": {"Var Confidence Experiment": 0.0, "RMS Slider End": 0.5},    # dropped
            "p3": {"Var Confidence Experiment": 0.02, "RMS Slider End": 1.0},   # dropped
            "p4": {"Var Confidence Experiment": 0.02, "RMS Slider End": 0.0},   # dropped
        })
        result = participants.filter_by_slider_values(
            df, var_cutoff=0.0, median_cutoff=0.0
        )
        assert set(result.index) == {"p1"}


class TestFilterByCatchAccuracy:
    def test_keeps_participants_meeting_cutoff(self):
        df = _stats_frame({
            "p1": {"Accuracy Catch": 1.0},
            "p2": {"Accuracy Catch": 0.8},
            "p3": {"Accuracy Catch": 0.5},
        })
        result = participants.filter_by_catch_accuracy(df, cutoff=0.8)
        assert set(result.index) == {"p1", "p2"}

    def test_requires_accuracy_catch_column(self):
        df = _stats_frame({"p1": {"Some Other Column": 1.0}})
        with pytest.raises(ValueError):
            participants.filter_by_catch_accuracy(df, cutoff=0.8)


class TestCatchFastMeasurablyChangesCatchAccuracyFilter:
    """End-to-end pin of correction 2: whether the fast-catch relabeling
    actually took effect changes a participant's catch accuracy, and
    therefore changes whether that participant survives the final filter
    stage — using only synthetic data, no real video files.
    """

    def _catch_response_accuracy(self, trial_metadata: pd.DataFrame) -> float:
        df_response_blocks = pd.DataFrame(
            {"response": [1, 2], "correct_response": [2, 2]},
            index=pd.Index(["C1", "C2"], name=_VIDEO_ID),
        )
        catch_only = trial_metadata[trial_metadata["trial"] == "Catch"]
        catch_responses = participants.filter_video_indexed_data(
            df_response_blocks, catch_only
        )
        return participants.response_accuracy(catch_responses)

    def test_reclassification_flips_participant_survival(self):
        df = _trial_metadata([
            {_VIDEO_ID: "C1", "trial": "Catch"},  # fast, and answered wrong
            {_VIDEO_ID: "C2", "trial": "Catch"},  # slow, and answered right
        ])
        timesteps = pd.Series({"C1": 2, "C2": 20})

        buggy_accuracy = self._catch_response_accuracy(df)

        reclassified = participants.reclassify_fast_catch_trials(
            df, timesteps, min_timesteps=8
        )
        fixed_accuracy = self._catch_response_accuracy(reclassified)

        assert buggy_accuracy == pytest.approx(0.5)
        assert fixed_accuracy == pytest.approx(1.0)

        stats_buggy = _stats_frame({"p1": {"Accuracy Catch": buggy_accuracy}})
        stats_fixed = _stats_frame({"p1": {"Accuracy Catch": fixed_accuracy}})

        assert len(participants.filter_by_catch_accuracy(stats_buggy, cutoff=0.8)) == 0
        assert len(participants.filter_by_catch_accuracy(stats_fixed, cutoff=0.8)) == 1


# ---------------------------------------------------------------------------
# Confidence-weighted choice
# ---------------------------------------------------------------------------

class TestComputeParticipantCwc:
    def _metadata(self):
        return _trial_metadata([
            # response equals color_entered -> Choice == -1
            {_VIDEO_ID: "V1", "color_entered": 1, "color_next": 2, "color_after_next": 3, "correct_coded": 1},
            # response equals color_next -> Choice == +1 (the non-color_entered default)
            {_VIDEO_ID: "V2", "color_entered": 1, "color_next": 2, "color_after_next": 3, "correct_coded": -1},
            # response equals color_after_next -> Choice == +1 as well: only
            # equality with color_entered flips Choice to -1, every other
            # response leaves it at the +1 default.
            {_VIDEO_ID: "V3", "color_entered": 1, "color_next": 2, "color_after_next": 3, "correct_coded": 1},
        ])

    def test_choice_sign_convention(self):
        responses = pd.DataFrame(
            {"response": [1, 2, 3], "slider_start": [0.0, 10.0, 20.0], "slider_end": [30.0, 60.0, 90.0]},
            index=pd.Index(["V1", "V2", "V3"], name=_VIDEO_ID),
        )
        result = participants.compute_participant_cwc(responses, self._metadata())
        assert result.loc["V1", "Choice"] == -1
        assert result.loc["V2", "Choice"] == 1
        assert result.loc["V3", "Choice"] == 1

    def test_cwc_is_choice_times_confidence_and_bounded(self):
        responses = pd.DataFrame(
            {"response": [1, 2, 3], "slider_start": [0.0, 10.0, 20.0], "slider_end": [30.0, 60.0, 90.0]},
            index=pd.Index(["V1", "V2", "V3"], name=_VIDEO_ID),
        )
        result = participants.compute_participant_cwc(responses, self._metadata())
        pd.testing.assert_series_equal(
            result["CWC"],
            result["Choice"] * result["Confidence"],
            check_names=False,
        )
        assert (result["CWC"] >= -1).all()
        assert (result["CWC"] <= 1).all()

    def test_confidence_adjusted_is_slope_residual_recentered(self):
        slider_start = np.array([0.0, 10.0, 20.0, 30.0])
        slider_end = np.array([5.0, 25.0, 35.0, 65.0])
        responses = pd.DataFrame(
            {
                "response": [1, 1, 1, 1],
                "slider_start": slider_start,
                "slider_end": slider_end,
            },
            index=pd.Index(["V1", "V1b", "V1c", "V1d"], name=_VIDEO_ID),
        )
        metadata = _trial_metadata([
            {_VIDEO_ID: vid, "color_entered": 1, "color_next": 2, "color_after_next": 3, "correct_coded": 1}
            for vid in ["V1", "V1b", "V1c", "V1d"]
        ])
        result = participants.compute_participant_cwc(responses, metadata)

        slope, intercept, *_ = linregress(slider_start, slider_end)
        expected = slider_end - (slope * slider_start + intercept) + slider_end.mean()
        np.testing.assert_allclose(result["Confidence Adjusted"].to_numpy(), expected)

    def test_low_confidence_range_falls_back_to_full_confidence(self):
        # confidence_range = sqrt(max**2 - min**2); with a tight cluster of
        # adjusted-confidence values this stays under the 70-point cutoff,
        # so every trial gets full (1.0) confidence rather than a percentile
        # normalization. slider_start must vary across trials -- the residual
        # regression is undefined for a constant predictor.
        responses = pd.DataFrame(
            {
                "response": [1, 1, 1],
                "slider_start": [10.0, 12.0, 14.0],
                "slider_end": [10.0, 11.0, 9.0],
            },
            index=pd.Index(["V1", "V2", "V3"], name=_VIDEO_ID),
        )
        metadata = _trial_metadata([
            {_VIDEO_ID: vid, "color_entered": 1, "color_next": 2, "color_after_next": 3, "correct_coded": 1}
            for vid in ["V1", "V2", "V3"]
        ])
        result = participants.compute_participant_cwc(responses, metadata)
        assert (result["Confidence"] == 1.0).all()

    def test_drops_rows_with_any_missing_value(self):
        responses = pd.DataFrame(
            {
                "response": [1, np.nan, 3],
                "slider_start": [0.0, 10.0, 20.0],
                "slider_end": [30.0, 60.0, 90.0],
            },
            index=pd.Index(["V1", "V2", "V3"], name=_VIDEO_ID),
        )
        result = participants.compute_participant_cwc(responses, self._metadata())
        assert "V2" not in result.index

    def test_restricts_to_intersection_with_trial_metadata(self):
        responses = pd.DataFrame(
            {
                "response": [1, 2],
                "slider_start": [0.0, 10.0],
                "slider_end": [30.0, 60.0],
            },
            index=pd.Index(["V1", "V_UNKNOWN"], name=_VIDEO_ID),
        )
        result = participants.compute_participant_cwc(responses, self._metadata())
        assert list(result.index) == ["V1"]


# ---------------------------------------------------------------------------
# Correction 1 — per-participant hazard/contingency CWC grouping
# ---------------------------------------------------------------------------

class TestParticipantCwcByHazard:
    def test_output_schema(self):
        cwc = {"p1": _participant_cwc_frame({"S1": 0.5, "S2": -0.5, "S3": 0.1, "S4": 0.2})}
        result = participants.participant_cwc_by_hazard(cwc, _straight_metadata())
        assert set(result.columns) >= {"Hazard Rate", "Grayzone Position", _PARTICIPANT_ID, "CWC"}

    def test_cwc_is_mean_over_matching_trials(self):
        cwc = {"p1": _participant_cwc_frame({"S1": 0.5, "S2": -0.5, "S3": 1.0, "S4": 1.0})}
        result = participants.participant_cwc_by_hazard(cwc, _straight_metadata())
        row = result[
            (result["Hazard Rate"] == "Low") & (result["Grayzone Position"] == 0)
        ]
        assert row["CWC"].iloc[0] == pytest.approx(0.5)

    def test_multiple_trials_in_same_bucket_are_averaged(self):
        metadata = _trial_metadata([
            {_VIDEO_ID: "S1", "trial": "Straight", "Hazard Rate": "Low", "idx_time": 0},
            {_VIDEO_ID: "S2", "trial": "Straight", "Hazard Rate": "Low", "idx_time": 0},
        ])
        cwc = {"p1": _participant_cwc_frame({"S1": 0.2, "S2": 0.8})}
        result = participants.participant_cwc_by_hazard(cwc, metadata)
        assert result["CWC"].iloc[0] == pytest.approx(0.5)

    def test_participant_grouping_is_invariant_to_other_participants(self):
        """Correction 1: participant p1's grouped rows must not depend on
        whether participant p2 is also present in the input.
        """
        cwc_p1_only = {"p1": _participant_cwc_frame({"S1": 0.5, "S2": -0.5, "S3": 0.1, "S4": 0.2})}
        cwc_both = {
            "p1": _participant_cwc_frame({"S1": 0.5, "S2": -0.5, "S3": 0.1, "S4": 0.2}),
            # p2 has a completely different, non-overlapping response set.
            "p2": _participant_cwc_frame({"S1": -0.9, "S3": -0.9}),
        }
        result_solo = participants.participant_cwc_by_hazard(cwc_p1_only, _straight_metadata())
        result_joint = participants.participant_cwc_by_hazard(cwc_both, _straight_metadata())

        p1_solo = result_solo[result_solo[_PARTICIPANT_ID] == "p1"].sort_values(
            ["Hazard Rate", "Grayzone Position"]
        ).reset_index(drop=True)
        p1_joint = result_joint[result_joint[_PARTICIPANT_ID] == "p1"].sort_values(
            ["Hazard Rate", "Grayzone Position"]
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(p1_solo, p1_joint)

        # The buckets come from the full metadata table, so a participant
        # missing trials still gets one row per bucket -- empty ones carry a
        # missing CWC rather than vanishing.
        p2_joint = result_joint[result_joint[_PARTICIPANT_ID] == "p2"]
        assert len(p2_joint) == len(p1_joint) == 4
        assert p2_joint["CWC"].isna().sum() == 2

    def test_is_deterministic(self):
        cwc = {"p1": _participant_cwc_frame({"S1": 0.5, "S2": -0.5, "S3": 0.1, "S4": 0.2})}
        first = participants.participant_cwc_by_hazard(cwc, _straight_metadata())
        second = participants.participant_cwc_by_hazard(cwc, _straight_metadata())
        pd.testing.assert_frame_equal(
            first.sort_values(list(first.columns)).reset_index(drop=True),
            second.sort_values(list(second.columns)).reset_index(drop=True),
        )


class TestParticipantCwcByContingency:
    def test_output_schema(self):
        cwc = {"p1": _participant_cwc_frame({"B1": 0.5, "B2": -0.5, "B3": 0.1})}
        result = participants.participant_cwc_by_contingency(cwc, _contingency_metadata())
        assert set(result.columns) >= {"Contingency", _PARTICIPANT_ID, "CWC", "Trial"}
        assert (result["Trial"] == "Bounce").all()

    def test_cwc_is_mean_over_matching_trials(self):
        cwc = {"p1": _participant_cwc_frame({"B1": 0.5, "B2": -0.5, "B3": 0.1})}
        result = participants.participant_cwc_by_contingency(cwc, _contingency_metadata())
        row = result[result["Contingency"] == "Low"]
        assert row["CWC"].iloc[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Artifact pipeline — requires the real data directory
# ---------------------------------------------------------------------------

def _skip_if_data_absent():
    if not _RAW_PARTICIPANT_DIR.exists() or not _PARTICIPANT_DATASET_DIR.exists():
        pytest.skip(
            "real participant data or cached trial metadata not present under "
            f"{_RAW_DATASET_DIR} / {_PARTICIPANT_DATASET_DIR}"
        )


@pytest.mark.slow
@pytest.mark.integration
class TestArtifactPipeline:
    @pytest.fixture(scope="class")
    def pipeline_output(self, tmp_path_factory):
        _skip_if_data_absent()
        output_dir = tmp_path_factory.mktemp("participant_artifacts")
        counts = participants.write_participant_stats(
            output_dir=output_dir,
            raw_dir=_RAW_DATASET_DIR,
            participant_dataset_dir=_PARTICIPANT_DATASET_DIR,
        )
        return output_dir, counts

    def test_writes_all_three_artifacts(self, pipeline_output):
        output_dir, _ = pipeline_output
        assert (output_dir / "participant_stats_filtered.parquet").exists()
        assert (output_dir / "participant_cwc.parquet").exists()
        assert (output_dir / "participant_counts.json").exists()

    def test_stats_filtered_schema(self, pipeline_output):
        output_dir, _ = pipeline_output
        df = pd.read_parquet(output_dir / "participant_stats_filtered.parquet")
        assert df.index.name == _PARTICIPANT_ID
        assert "Accuracy Catch" in df.columns
        assert len(df) > 0

    def test_cwc_schema(self, pipeline_output):
        output_dir, _ = pipeline_output
        df = pd.read_parquet(output_dir / "participant_cwc.parquet")
        assert set(df.columns) >= {
            _PARTICIPANT_ID, _VIDEO_ID, "Choice", "Confidence", "CWC", "correct_coded",
        }
        assert len(df) > 0

    def test_counts_json_schema_and_bounds(self, pipeline_output):
        output_dir, counts = pipeline_output
        with open(output_dir / "participant_counts.json") as f:
            on_disk = json.load(f)
        assert on_disk == counts

        assert "total_loaded" in counts
        assert "stage_counts" in counts
        assert "final_n" in counts

        total_loaded = counts["total_loaded"]
        final_n = counts["final_n"]
        assert final_n > 0
        assert final_n <= total_loaded

        stage_counts = counts["stage_counts"]
        assert isinstance(stage_counts, dict)
        assert len(stage_counts) > 0
        values = list(stage_counts.values())
        # Each stage can only remove participants, never add them.
        assert all(a >= b for a, b in zip(values, values[1:], strict=False))
        assert values[0] <= total_loaded
        assert values[-1] == final_n


@pytest.mark.slow
@pytest.mark.integration
class TestPipelineDeterminism:
    def test_two_runs_produce_identical_artifacts(self, tmp_path_factory):
        _skip_if_data_absent()
        out_a = tmp_path_factory.mktemp("participant_artifacts_a")
        out_b = tmp_path_factory.mktemp("participant_artifacts_b")

        counts_a = participants.write_participant_stats(
            output_dir=out_a,
            raw_dir=_RAW_DATASET_DIR,
            participant_dataset_dir=_PARTICIPANT_DATASET_DIR,
        )
        counts_b = participants.write_participant_stats(
            output_dir=out_b,
            raw_dir=_RAW_DATASET_DIR,
            participant_dataset_dir=_PARTICIPANT_DATASET_DIR,
        )

        assert counts_a == counts_b

        stats_a = pd.read_parquet(out_a / "participant_stats_filtered.parquet")
        stats_b = pd.read_parquet(out_b / "participant_stats_filtered.parquet")
        pd.testing.assert_frame_equal(stats_a, stats_b)

        cwc_a = pd.read_parquet(out_a / "participant_cwc.parquet")
        cwc_b = pd.read_parquet(out_b / "participant_cwc.parquet")
        pd.testing.assert_frame_equal(cwc_a, cwc_b)

        assert (
            (out_a / "participant_stats_filtered.parquet").read_bytes()
            == (out_b / "participant_stats_filtered.parquet").read_bytes()
        )

"""Contract tests for the model-CWC transforms.

Pins the interface of four transforms that live in
``src/learning_in_context/visualization/transforms.py`` alongside the existing
fig4/fig5/fig6/fig7 transforms, following the same keying discipline: every
function is keyed on a dataset name plus small hashable params, loads the
underlying ``.npz``/``.csv`` artifacts under ``data/cache/model_states/``
internally, and is memoized through the shared ``transforms.MEMORY`` instance.

The four transforms:

* ``trial_metadata(dataset)`` — the per-video trial table (``trial_meta.csv``
  plus the Catch-Fast reclassification: a handful of "Catch" trials whose
  final color was already stable well before the trial ended are relabeled
  "Catch-Fast", and the two hazard/contingency condition columns become
  ordered categoricals).

* ``model_cwc_frame(dataset, model_types, num_participants, seed)`` — one row
  per (model, sampled choice, video). For every prediction file under
  ``data/cache/model_states/<dataset>/<model_type>/``, a per-video color
  distribution at the trial's final timestep is sampled some number of times;
  each sample yields a confidence-weighted-choice value ``cwc = choice_coded *
  choice_prob``, where ``choice_coded`` is -1 when the sampled color equals
  the color the participant actually entered for that video and +1
  otherwise, and ``choice_prob`` is the model's probability mass on the
  sampled color. Sampling must be seeded — a bare ``np.random.rand`` call with
  no seed is exactly the kind of irreproducibility this transform exists to
  close off.

* ``model_cwc_by_hazard`` / ``model_cwc_by_contingency`` — mean-``cwc``
  aggregates of ``model_cwc_frame`` restricted to the Straight and Bounce
  subsets respectively, grouped by the condition columns the hazard/
  contingency panels split on.

Interface-only tests (no cache-dir dependency) run under the default
selection. Tests that load the real cached model-state artifacts are marked
``slow`` + ``integration``, mirroring ``tests/test_fig5_panels.py`` /
``tests/test_fig4_panels.py``.

Resolved ambiguities (see module-level notes below the imports):

1. Parameter naming follows the existing ``transforms.py`` house style
   (``dataset: str`` identifying a subdirectory of ``data/cache/model_states/``,
   e.g. ``"participant_dataset"``) rather than a raw ``cache_dir`` path, for
   consistency with every other transform in the module.
2. ``model_cwc_by_hazard`` / ``model_cwc_by_contingency`` carry a ``model``
   column in addition to ``model_sample``/condition columns/``cwc`` — needed
   to keep RNN, LSTM, and IBO rows distinguishable in one frame when
   ``model_types`` names more than one model. The source notebook cell loops
   over model type and renders one panel per iteration, so a caller can still
   filter to one model before rendering.
3. ``num_participants`` is required and explicit here (not resolved via the
   participant-side artifacts, which are a separate, not-yet-landed port
   item) — these tests never rely on a default/None-triggered lookup.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    from learning_in_context.visualization import transforms
    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - only if transforms.py itself breaks
    transforms = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASET = "participant_dataset"
_DATASET_DIR = _REPO_ROOT / "data" / "cache" / "model_states" / _DATASET
_RAW_TRIAL_META = _DATASET_DIR / "trial_meta.csv"

_HAS_REAL_DATA = _RAW_TRIAL_META.exists()

_FORBIDDEN_ARRAY_PARAM_NAMES = {
    "states",
    "samples",
    "targets",
    "preds",
    "df_selected",
    "df_data",
    "df",
    "array",
    "arrays",
    "data",
}


def test_transforms_module_is_importable():
    if _IMPORT_ERROR is not None:
        pytest.fail(
            "learning_in_context.visualization.transforms is not importable: "
            f"{_IMPORT_ERROR!r}"
        )


def _assert_memoized_with_shared_memory(fn, name: str) -> None:
    assert hasattr(fn, "func") and hasattr(fn, "store_backend"), (
        f"{name} does not look like a joblib-memoized function"
    )
    mem_root = Path(transforms.MEMORY.location).resolve()
    fn_loc = Path(fn.store_backend.location).resolve()
    assert fn_loc == mem_root or mem_root in fn_loc.parents, (
        f"{name} is memoized with a Memory rooted at {fn_loc}, not the shared "
        f"transforms.MEMORY at {mem_root}"
    )


# --------------------------------------------------------------------------- #
# Interface contracts — no real cache dir required
# --------------------------------------------------------------------------- #
class TestTrialMetadataSignature:
    def test_exists_and_is_callable(self):
        assert hasattr(transforms, "trial_metadata")
        assert callable(transforms.trial_metadata)

    def test_no_raw_array_parameters(self):
        sig = inspect.signature(transforms.trial_metadata)
        offending = set(sig.parameters) & _FORBIDDEN_ARRAY_PARAM_NAMES
        assert not offending, (
            f"trial_metadata takes raw-array-shaped params {offending}; it must "
            "key on a dataset name and load data internally."
        )

    def test_is_memoized_with_the_shared_memory_instance(self):
        _assert_memoized_with_shared_memory(transforms.trial_metadata, "trial_metadata")


class TestModelCwcFrameSignature:
    def test_exists_and_is_callable(self):
        assert hasattr(transforms, "model_cwc_frame")
        assert callable(transforms.model_cwc_frame)

    def test_no_raw_array_parameters(self):
        sig = inspect.signature(transforms.model_cwc_frame)
        offending = set(sig.parameters) & _FORBIDDEN_ARRAY_PARAM_NAMES
        assert not offending, (
            f"model_cwc_frame takes raw-array-shaped params {offending}; it must "
            "load prediction arrays internally, keyed on dataset/model_types/ids."
        )

    def test_seed_parameter_is_required(self):
        """The sampling step must not fall back to an unseeded RNG.

        A ``seed`` parameter with no default forces every call site to state
        its seed explicitly, closing off the source notebook's bare
        ``np.random.rand()`` (irreproducible run to run).
        """
        sig = inspect.signature(transforms.model_cwc_frame)
        assert "seed" in sig.parameters, "model_cwc_frame has no seed parameter"
        assert sig.parameters["seed"].default is inspect.Parameter.empty, (
            "model_cwc_frame's seed parameter has a default — it must be "
            "required so sampling can never silently run unseeded."
        )

    def test_is_memoized_with_the_shared_memory_instance(self):
        _assert_memoized_with_shared_memory(transforms.model_cwc_frame, "model_cwc_frame")


class TestModelCwcByHazardSignature:
    def test_exists_and_is_callable(self):
        assert hasattr(transforms, "model_cwc_by_hazard")
        assert callable(transforms.model_cwc_by_hazard)

    def test_no_raw_array_parameters(self):
        sig = inspect.signature(transforms.model_cwc_by_hazard)
        offending = set(sig.parameters) & _FORBIDDEN_ARRAY_PARAM_NAMES
        assert not offending

    def test_seed_parameter_is_required(self):
        sig = inspect.signature(transforms.model_cwc_by_hazard)
        assert "seed" in sig.parameters
        assert sig.parameters["seed"].default is inspect.Parameter.empty

    def test_is_memoized_with_the_shared_memory_instance(self):
        _assert_memoized_with_shared_memory(
            transforms.model_cwc_by_hazard, "model_cwc_by_hazard"
        )


class TestModelCwcByContingencySignature:
    def test_exists_and_is_callable(self):
        assert hasattr(transforms, "model_cwc_by_contingency")
        assert callable(transforms.model_cwc_by_contingency)

    def test_no_raw_array_parameters(self):
        sig = inspect.signature(transforms.model_cwc_by_contingency)
        offending = set(sig.parameters) & _FORBIDDEN_ARRAY_PARAM_NAMES
        assert not offending

    def test_seed_parameter_is_required(self):
        sig = inspect.signature(transforms.model_cwc_by_contingency)
        assert "seed" in sig.parameters
        assert sig.parameters["seed"].default is inspect.Parameter.empty

    def test_is_memoized_with_the_shared_memory_instance(self):
        _assert_memoized_with_shared_memory(
            transforms.model_cwc_by_contingency, "model_cwc_by_contingency"
        )


# --------------------------------------------------------------------------- #
# Behavioral contracts — require the real cached artifacts under
# data/cache/model_states/participant_dataset/ (trial_meta.csv, the video
# tree for the Catch-Fast reclassification, and the ibo/rnn/lstm .npz preds).
# --------------------------------------------------------------------------- #
_needs_real_data = pytest.mark.skipif(
    not _HAS_REAL_DATA,
    reason=f"real trial_meta.csv not found under {_DATASET_DIR}",
)


@pytest.mark.slow
@pytest.mark.integration
@_needs_real_data
class TestTrialMetadata:
    def _raw_trial_meta(self) -> pd.DataFrame:
        return pd.read_csv(_RAW_TRIAL_META, index_col=0)

    def test_row_count_matches_raw_trial_meta(self):
        df = transforms.trial_metadata(_DATASET)
        raw = self._raw_trial_meta()
        assert len(df) == len(raw)

    def test_has_expected_columns(self):
        df = transforms.trial_metadata(_DATASET)
        required = {
            "trial",
            "Hazard Rate",
            "Contingency",
            "idx_time",
            "correct_response",
            "length",
        }
        missing = required - set(df.columns)
        assert not missing, f"trial_metadata is missing columns: {missing}"

    def test_hazard_rate_is_ordered_categorical(self):
        df = transforms.trial_metadata(_DATASET)
        hz = df["Hazard Rate"]
        assert isinstance(hz.dtype, pd.CategoricalDtype)
        assert hz.dtype.ordered
        assert list(hz.dtype.categories) == ["Low", "High"]

    def test_contingency_is_ordered_categorical(self):
        df = transforms.trial_metadata(_DATASET)
        cont = df["Contingency"]
        assert isinstance(cont.dtype, pd.CategoricalDtype)
        assert cont.dtype.ordered
        assert list(cont.dtype.categories) == ["Low", "Medium", "High"]

    def test_trial_values_are_within_known_vocabulary(self):
        df = transforms.trial_metadata(_DATASET)
        known = {"Straight", "Bounce", "Catch", "Catch-Fast"}
        observed = set(df["trial"].unique())
        assert observed <= known, f"unexpected trial labels: {observed - known}"

    def test_catch_fast_reclassification_is_nonempty_and_conserves_catch_count(self):
        """Guards the pandas chained-assignment no-op documented for this port.

        ``df["trial"][idx_catch_skip] = "Catch-Fast"`` is a silent no-op under
        copy-on-write pandas; a correct port relabels at least one row without
        changing how many rows started out as some flavor of "Catch".
        """
        df = transforms.trial_metadata(_DATASET)
        raw = self._raw_trial_meta()

        raw_catch_count = (raw["trial"] == "Catch").sum()
        assert raw_catch_count > 0, "fixture assumption: raw data has Catch trials"

        catch_count = (df["trial"] == "Catch").sum()
        catch_fast_count = (df["trial"] == "Catch-Fast").sum()

        assert catch_fast_count > 0, (
            "trial_metadata reclassified zero Catch trials to Catch-Fast — "
            "looks like the pandas-3 chained-assignment no-op"
        )
        assert catch_count + catch_fast_count == raw_catch_count

    def test_other_trial_types_are_unaffected_by_reclassification(self):
        df = transforms.trial_metadata(_DATASET)
        raw = self._raw_trial_meta()
        for label in ("Straight", "Bounce"):
            assert (df["trial"] == label).sum() == (raw["trial"] == label).sum()

    def test_memoized_result_matches_a_fresh_recompute(self):
        """The cached frame must equal what the undecorated function computes.

        Calling the transform twice would be satisfied by a cache hit alone, so
        the second call goes through ``.func`` — joblib's handle on the
        original, uncached function.
        """
        cached = transforms.trial_metadata(_DATASET)
        fresh = transforms.trial_metadata.func(_DATASET)
        pd.testing.assert_frame_equal(cached, fresh)


@pytest.mark.slow
@pytest.mark.integration
@_needs_real_data
class TestModelCwcFrame:
    def test_required_schema_columns_present(self):
        df = transforms.model_cwc_frame(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )
        required = {
            "model",
            "sample_id",
            "cwc",
            "choice_prob",
            "Hazard Rate",
            "Contingency",
            "Grayzone Position",
            "trial",
        }
        missing = required - set(df.columns)
        assert not missing, f"model_cwc_frame is missing columns: {missing}"

    def test_model_labels_are_uppercased(self):
        df = transforms.model_cwc_frame(
            _DATASET, model_types=("ibo", "rnn"), num_participants=5, seed=0
        )
        assert set(df["model"].unique()) <= {"IBO", "RNN", "LSTM"}
        assert set(df["model"].unique()) == {"IBO", "RNN"}

    def test_cwc_and_choice_prob_are_in_range(self):
        df = transforms.model_cwc_frame(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )
        assert (df["cwc"] >= -1).all() and (df["cwc"] <= 1).all()
        assert (df["choice_prob"] >= 0).all() and (df["choice_prob"] <= 1).all()

    def test_cwc_sign_convention_matches_choice_coded_times_choice_prob(self):
        """cwc == choice_coded * choice_prob, choice_coded in {-1, +1}.

        Magnitude alone (|cwc| == choice_prob) would pass with the sign
        inverted, so the sign itself is pinned against the entered colour:
        ``choice_coded`` is -1 exactly where the sampled colour matches the
        colour entered for that video, +1 everywhere else.
        """
        df = transforms.model_cwc_frame(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )
        assert set(np.unique(df["choice_coded"].to_numpy())) <= {-1, 1}
        np.testing.assert_allclose(
            df["cwc"].to_numpy(),
            df["choice_coded"].to_numpy() * df["choice_prob"].to_numpy(),
        )

        meta = transforms.trial_metadata(_DATASET)
        # The trial table stores colours 1-based; predictions are 0-based.
        color_entered = meta["color_entered"].reindex(df["Video ID"].to_numpy()) - 1
        stays = df["choice_sampled"].to_numpy() == color_entered.to_numpy()
        expected_coded = np.where(stays, -1, 1)
        assert stays.any() and (~stays).any(), (
            "fixture assumption: both stay and switch responses are sampled"
        )
        np.testing.assert_array_equal(df["choice_coded"].to_numpy(), expected_coded)

    def test_choice_prob_is_the_sampled_colors_probability(self):
        """choice_prob must read the model's mass on the colour it sampled."""
        df = transforms.model_cwc_frame(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )
        probs = df[["prob_r", "prob_g", "prob_b"]].to_numpy()
        sampled = probs[np.arange(len(df)), df["choice_sampled"].to_numpy()]
        np.testing.assert_allclose(sampled, df["choice_prob"].to_numpy())

    def test_one_row_per_sample_and_video(self):
        df = transforms.model_cwc_frame(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )
        num_videos = len(transforms.trial_metadata(_DATASET))
        assert not df.duplicated(subset=["sample_id", "Video ID"]).any()
        assert len(df) == num_videos * df["sample_id"].nunique()

    def test_sample_count_follows_ceil_division_of_num_participants(self):
        """num_samples per prediction file == num_participants // n_files + 1.

        The real ``rnn`` directory under ``participant_dataset`` holds exactly
        10 prediction files, so with num_participants=25 each file should
        contribute 25 // 10 + 1 == 3 samples, for 30 distinct sample ids.
        """
        n_files = len(list((_DATASET_DIR / "rnn").glob("*.npz")))
        assert n_files == 10, "fixture assumption: 10 rnn prediction files"

        num_participants = 25
        expected_samples_per_file = num_participants // n_files + 1

        df = transforms.model_cwc_frame(
            _DATASET, model_types=("rnn",), num_participants=num_participants, seed=0
        )
        counts_per_exp = df.groupby("exp_id")["sample_id"].nunique()
        assert (counts_per_exp == expected_samples_per_file).all()
        assert df["sample_id"].nunique() == n_files * expected_samples_per_file

    def test_same_seed_yields_identical_frame(self):
        """Re-running with the same seed must reproduce the frame exactly.

        The second call goes through ``.func`` (joblib's handle on the original,
        uncached function) so the check exercises the sampling, not the cache.
        """
        first = transforms.model_cwc_frame(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )
        second = transforms.model_cwc_frame.func(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )
        pd.testing.assert_frame_equal(first, second)

    def test_different_seed_yields_different_sampled_choices(self):
        seed0 = transforms.model_cwc_frame(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )
        seed1 = transforms.model_cwc_frame(
            _DATASET, model_types=("ibo",), num_participants=5, seed=1
        )
        assert not seed0["choice_sampled"].equals(seed1["choice_sampled"])

    def test_seeding_is_stable_under_model_types_narrowing(self):
        """A single sample's rows shouldn't reshuffle when other model types
        are added to or removed from ``model_types`` — seeding must be derived
        per (model_type, exp_id, sample_index), not from one loop-top seed.
        """
        rnn_dir = _DATASET_DIR / "rnn"
        one_exp_id = sorted(p.stem for p in rnn_dir.glob("*.npz"))[0]

        combined = transforms.model_cwc_frame(
            _DATASET, model_types=("ibo", "rnn"), num_participants=5, seed=0
        )
        solo = transforms.model_cwc_frame(
            _DATASET, model_types=("rnn",), num_participants=5, seed=0
        )

        combined_rows = (
            combined[(combined["exp_id"] == one_exp_id) & (combined["sample_id"] == f"{one_exp_id}-0")]
            .sort_values("Video ID")
            .reset_index(drop=True)
        )
        solo_rows = (
            solo[(solo["exp_id"] == one_exp_id) & (solo["sample_id"] == f"{one_exp_id}-0")]
            .sort_values("Video ID")
            .reset_index(drop=True)
        )
        assert len(combined_rows) > 0 and len(solo_rows) > 0
        np.testing.assert_array_equal(
            combined_rows["choice_sampled"].to_numpy(), solo_rows["choice_sampled"].to_numpy()
        )


@pytest.mark.slow
@pytest.mark.integration
@_needs_real_data
class TestModelCwcByHazard:
    def test_columns_are_exactly_the_tidy_hazard_schema(self):
        df = transforms.model_cwc_by_hazard(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )
        assert set(df.columns) == {"model", "model_sample", "Hazard Rate", "Grayzone Position", "cwc"}

    def test_values_are_within_cwc_range(self):
        df = transforms.model_cwc_by_hazard(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )
        assert (df["cwc"] >= -1).all() and (df["cwc"] <= 1).all()

    def test_matches_independent_groupby_of_model_cwc_frame_on_straight_trials(self):
        """Reconstructs the grouping semantics directly from model_cwc_frame,
        mirroring per-(model, sample, Hazard Rate, Grayzone Position) mean cwc
        over Straight trials only, and checks the transform agrees.
        """
        base = transforms.model_cwc_frame(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )
        straight = base[base["trial"] == "Straight"]
        expected = (
            straight.groupby(["model", "sample_id", "Hazard Rate", "Grayzone Position"], observed=True)["cwc"]
            .mean()
            .reset_index()
            .rename(columns={"sample_id": "model_sample"})
        )

        actual = transforms.model_cwc_by_hazard(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )

        assert len(actual) == len(expected)
        merged = expected.merge(
            actual,
            on=["model", "model_sample", "Hazard Rate", "Grayzone Position"],
            suffixes=("_expected", "_actual"),
        )
        assert len(merged) == len(expected), "row keys did not line up between the two groupings"
        np.testing.assert_allclose(
            merged["cwc_expected"].to_numpy(), merged["cwc_actual"].to_numpy()
        )

    def test_only_straight_trials_contribute(self):
        """Non-straight trials must be excluded, not merged into the aggregate.

        Bounce and catch trials carry no grayzone position (the column reads
        -1 for them), so admitting any of them would both add rows and shift
        the means of the cells they merged into.
        """
        by_hazard = transforms.model_cwc_by_hazard(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )
        base = transforms.model_cwc_frame(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )
        straight_grayzones = set(
            base.loc[base["trial"] == "Straight", "Grayzone Position"].unique()
        )
        non_straight_grayzones = set(
            base.loc[base["trial"] != "Straight", "Grayzone Position"].unique()
        )
        assert non_straight_grayzones - straight_grayzones, (
            "fixture assumption: non-straight trials carry a distinct grayzone value"
        )
        observed_grayzones = set(by_hazard["Grayzone Position"].unique())
        assert observed_grayzones <= straight_grayzones
        assert not observed_grayzones & (non_straight_grayzones - straight_grayzones)

        # Every straight cell must survive: one row per
        # (sample, hazard rate, grayzone position) the straight trials cover.
        straight = base[base["trial"] == "Straight"]
        expected_rows = len(
            straight.groupby(
                ["model", "sample_id", "Hazard Rate", "Grayzone Position"],
                observed=True,
            )
        )
        assert len(by_hazard) == expected_rows


@pytest.mark.slow
@pytest.mark.integration
@_needs_real_data
class TestModelCwcByContingency:
    def test_columns_are_exactly_the_tidy_contingency_schema(self):
        df = transforms.model_cwc_by_contingency(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )
        assert set(df.columns) == {"model", "model_sample", "Contingency", "cwc"}

    def test_values_are_within_cwc_range(self):
        df = transforms.model_cwc_by_contingency(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )
        assert (df["cwc"] >= -1).all() and (df["cwc"] <= 1).all()

    def test_matches_independent_groupby_of_model_cwc_frame_on_bounce_trials(self):
        base = transforms.model_cwc_frame(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )
        bounce = base[base["trial"] == "Bounce"]
        expected = (
            bounce.groupby(["model", "sample_id", "Contingency"], observed=True)["cwc"]
            .mean()
            .reset_index()
            .rename(columns={"sample_id": "model_sample"})
        )

        actual = transforms.model_cwc_by_contingency(
            _DATASET, model_types=("ibo",), num_participants=5, seed=0
        )

        assert len(actual) == len(expected)
        merged = expected.merge(
            actual, on=["model", "model_sample", "Contingency"], suffixes=("_expected", "_actual")
        )
        assert len(merged) == len(expected), "row keys did not line up between the two groupings"
        np.testing.assert_allclose(
            merged["cwc_expected"].to_numpy(), merged["cwc_actual"].to_numpy()
        )

"""Contract tests for the ideal-observer belief-curve transform.

Pins the interface of ``ideal_observer_belief_curves``, which is expected to
live in ``src/learning_in_context/visualization/transforms.py`` alongside the
existing fig4-fig7 transforms and follow the same keying discipline: a
function keyed on a dataset name plus small hashable scalars, memoized
through the shared ``transforms.MEMORY`` instance, that loads the underlying
cached artifacts under ``data/cache/model_states/`` internally rather than
accepting arrays.

The transform's quantity: for one exemplar trial, the ideal Bayesian
observer's per-frame posterior probability that the ball's color now differs
from the color it last showed on a visible frame, computed by running the
observer's own belief update forward under a swept task parameter (hazard
rate for straight-path trials, contingency for wall-bounce trials) while
holding the trial's geometry fixed. On a visible frame this probability
equals the swept parameter exactly (the belief resets to the observed color
and takes one step); across occluded frames it evolves under repeated
belief-transitions and only diverges between contingency levels once a
wall-bounce event is reached, because the observer's own transition math
has no way for a level swept on the bounce-triggered parameter to affect a
frame away from a wall.

Interface-only tests (signature, memoization, no-raw-array-parameters,
static import-not-copy checks) do not need cached data and are not marked
slow. Tests that call the transform need the real ``control_dataset``
artifacts under ``data/cache/model_states/control_dataset/`` and are
skipped if those are absent; the heaviest tier, which checks the
transform's output against a documented parameter regime, is additionally
marked ``slow`` and ``integration``.
"""

from __future__ import annotations

import ast
import hashlib
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

try:
    from learning_in_context.models import ideal_observer as _ideal_observer_module

    _IDEAL_OBSERVER_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - only if the observer module breaks
    _ideal_observer_module = None  # type: ignore[assignment]
    _IDEAL_OBSERVER_IMPORT_ERROR = exc

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASET = "control_dataset"
_DATASET_DIR = _REPO_ROOT / "data" / "cache" / "model_states" / _DATASET
_HAS_REAL_DATA = (_DATASET_DIR / "trial_meta.csv").exists() and (
    _DATASET_DIR / "samples.npy"
).exists()

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

# Two-level hazard sweep and three-level contingency sweep, held fixed across
# the whole test module so every test exercises the same exemplar trial.
_HAZARD_LEVELS = (("High", 0.0245), ("Low", 0.0099))
_CONTINGENCY_LEVELS = (("High", 0.95), ("Medium", 0.5), ("Low", 0.05))
_LEAD_IN_FRAMES = 5
_LAST_VISIBLE_FRAME = _LEAD_IN_FRAMES - 1  # window coordinates: entry - 1

_EXPECTED_COLUMNS = {
    "level",
    "param",
    "frame",
    "p_change",
    "occluded",
    "is_bounce",
    "video_id",
    "endpoint_offset",
}

_EXPECTED_SIGNATURE_DEFAULTS = {
    "dataset": "control_dataset",
    "trial_type": "Straight",
    "sweep": "hazard",
    "levels": (("High", 0.0245), ("Low", 0.0099)),
    "fixed_pccnvc": 0.0245,
    "fixed_pccovc": 0.5,
    "pvc": 0.0,
    "idx_time": 2,
    "exemplar_rank": 0,
    "lead_in_frames": 5,
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
class TestIdealObserverBeliefCurvesSignature:
    def test_exists_and_is_callable(self):
        assert hasattr(transforms, "ideal_observer_belief_curves")
        assert callable(transforms.ideal_observer_belief_curves)

    def test_no_raw_array_parameters(self):
        sig = inspect.signature(transforms.ideal_observer_belief_curves)
        offending = set(sig.parameters) & _FORBIDDEN_ARRAY_PARAM_NAMES
        assert not offending, (
            f"ideal_observer_belief_curves takes raw-array-shaped params "
            f"{offending}; it must key on a dataset name and scalar params "
            "and load data internally."
        )

    def test_is_memoized_with_the_shared_memory_instance(self):
        _assert_memoized_with_shared_memory(
            transforms.ideal_observer_belief_curves, "ideal_observer_belief_curves"
        )

    def test_signature_has_the_documented_parameters_and_defaults(self):
        sig = inspect.signature(transforms.ideal_observer_belief_curves)
        params = sig.parameters
        missing = set(_EXPECTED_SIGNATURE_DEFAULTS) - set(params)
        assert not missing, (
            f"ideal_observer_belief_curves is missing expected parameters: {missing}"
        )
        for name, default in _EXPECTED_SIGNATURE_DEFAULTS.items():
            assert params[name].default == default, (
                f"{name} default is {params[name].default!r}, expected {default!r}"
            )

    def test_is_annotated_as_returning_a_dataframe(self):
        """The return annotation is part of the pinned contract, so a missing
        one is a failure rather than something to skip past: skipping here
        would let an unannotated (or differently typed) transform pass the
        interface tier silently.
        """
        sig = inspect.signature(transforms.ideal_observer_belief_curves)
        assert sig.return_annotation is not inspect.Signature.empty, (
            "ideal_observer_belief_curves has no return annotation; it must be "
            "annotated as returning pd.DataFrame"
        )
        assert sig.return_annotation in (pd.DataFrame, "pd.DataFrame")

    def test_has_a_keyword_only_observer_fingerprint_parameter(self):
        """Guards the observer against silent cache staleness.

        joblib keys ``ideal_observer_belief_curves``'s cache entries on the
        wrapper's own source plus its bound arguments; nothing about
        ``ideal_observer.py``'s source enters that key on its own. Threading
        a fingerprint of the observer module through as a keyword-only
        default closes that gap: ``filter_args`` binds defaults into the
        hashed argument dict, so an edit to the observer's semantics changes
        the fingerprint, which changes every call's cache key, which forces
        a recompute instead of silently re-serving a stale curve.
        """
        sig = inspect.signature(transforms.ideal_observer_belief_curves)
        params = sig.parameters
        assert "observer_fingerprint" in params, (
            "ideal_observer_belief_curves has no observer_fingerprint parameter"
        )
        observer_fingerprint = params["observer_fingerprint"]
        assert observer_fingerprint.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"observer_fingerprint is {observer_fingerprint.kind}, expected "
            "KEYWORD_ONLY so it cannot be passed positionally and shifted by "
            "an unrelated signature change"
        )
        assert observer_fingerprint.default is transforms.IDEAL_OBSERVER_FINGERPRINT
        assert observer_fingerprint.default == transforms.IDEAL_OBSERVER_FINGERPRINT

    def test_the_fingerprint_is_not_excluded_from_the_cache_key(self):
        """A fingerprint joblib is told to ignore guards nothing.

        ``Memory.cache(ignore=[...])`` drops the named arguments before the
        bound-argument dict is hashed, so listing ``observer_fingerprint``
        there would leave the parameter on the signature while restoring the
        exact staleness this feature exists to close. The behavioral check
        below needs the real ``control_dataset`` artifacts and is skipped
        without them; this one holds wherever the module imports.
        """
        fn = transforms.ideal_observer_belief_curves
        assert "observer_fingerprint" in inspect.signature(fn).parameters, (
            "ideal_observer_belief_curves has no observer_fingerprint "
            "parameter, so this check would pass vacuously"
        )
        assert "observer_fingerprint" not in (fn.ignore or []), (
            "observer_fingerprint is in the memoized function's ignore list, "
            "so joblib strips it before hashing and it never reaches the "
            "cache key"
        )

    def test_impl_helper_does_not_take_the_fingerprint(self):
        """The wrapper strips the fingerprint before delegating to the impl.

        The fingerprint exists only to participate in the joblib cache key;
        the impl helper's own signature is unchanged by this feature, so it
        must not gain the parameter too.
        """
        impl_sig = inspect.signature(transforms._ideal_observer_belief_curves_impl)
        assert "observer_fingerprint" not in impl_sig.parameters, (
            "_ideal_observer_belief_curves_impl must not take "
            "observer_fingerprint; the wrapper computes/consumes it and "
            "forwards only the original arguments"
        )


class TestIdealObserverFingerprint:
    """Pins the fingerprint's derivation, not just its presence.

    A fingerprint computed over the wrong scope (e.g. only the
    ``IdealBayesianObserver`` class, omitting the base class it inherits
    ``T`` from) would still satisfy the signature test above while failing
    to actually change when the semantics it is meant to guard change.
    """

    def test_fingerprint_exists_and_is_a_16_char_lowercase_hex_string(self):
        assert hasattr(transforms, "IDEAL_OBSERVER_FINGERPRINT")
        fingerprint = transforms.IDEAL_OBSERVER_FINGERPRINT
        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 16
        assert fingerprint == fingerprint.lower()
        int(fingerprint, 16)  # raises ValueError if not hex

    def test_fingerprint_equals_a_blake2b_8_digest_of_the_whole_observer_module(self):
        if _IDEAL_OBSERVER_IMPORT_ERROR is not None:
            pytest.fail(
                "learning_in_context.models.ideal_observer is not importable: "
                f"{_IDEAL_OBSERVER_IMPORT_ERROR!r}"
            )
        expected = hashlib.blake2b(
            inspect.getsource(_ideal_observer_module).encode("utf-8"), digest_size=8
        ).hexdigest()
        assert transforms.IDEAL_OBSERVER_FINGERPRINT == expected


# --------------------------------------------------------------------------- #
# Import-not-copy — the transform must call the real observer, not a
# reimplementation of its belief update.
# --------------------------------------------------------------------------- #
class TestImportNotCopy:
    """Negative assertions only bite once the transform exists.

    Every test here would pass against a module that simply has no
    ``ideal_observer_belief_curves`` in it, so each one first asserts the
    transform is present. Without that precondition the tier reports green
    before a line of the transform has been written.
    """

    @staticmethod
    def _transform_sources() -> str:
        """Source of the transform and of the helpers it delegates to.

        Scoped to the transform rather than to the whole module because
        ``transforms.py`` is shared with unrelated figure transforms: a
        module-wide grep would make this test fail on a sibling's code.
        """
        fn = transforms.ideal_observer_belief_curves
        target = getattr(fn, "func", fn)
        sources = [inspect.getsource(target)]
        for name, obj in vars(transforms).items():
            if name.startswith("_") and "belief_curve" in name and callable(obj):
                sources.append(inspect.getsource(obj))
        return "\n".join(sources)

    def test_module_imports_the_real_bayesian_observer(self):
        assert transforms is not None, _IMPORT_ERROR
        assert hasattr(transforms, "ideal_observer_belief_curves")
        source = inspect.getsource(transforms)
        assert "from learning_in_context.models.ideal_observer import" in source, (
            "transforms.py does not import learning_in_context.models.ideal_observer"
        )
        assert "IdealBayesianObserver" in source, (
            "transforms.py does not reference IdealBayesianObserver"
        )

    def test_module_defines_no_class_of_its_own_named_like_an_observer(self):
        assert transforms is not None, _IMPORT_ERROR
        assert hasattr(transforms, "ideal_observer_belief_curves")
        source = inspect.getsource(transforms)
        tree = ast.parse(source)
        observer_classes = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and "Observer" in node.name
        ]
        assert not observer_classes, (
            f"transforms.py defines its own observer-like class(es): "
            f"{observer_classes}; belief propagation must go through the "
            "imported model, not a local copy"
        )

    def test_transform_does_not_reimplement_a_transition_matrix(self):
        assert transforms is not None, _IMPORT_ERROR
        assert hasattr(transforms, "ideal_observer_belief_curves")
        source = self._transform_sources()
        # Guard the grep's own scope: if the helper discovery below stops
        # finding the propagation code (a rename, say), the grep would run
        # over a thin wrapper and pass vacuously.
        assert "IdealBayesianObserver" in source, (
            "the scoped source does not contain the observer call, so this "
            "grep is not looking at the belief-propagation code"
        )
        assert "torch.stack(" not in source, (
            "the belief-curve transform appears to build its own "
            "transition-matrix stack instead of delegating to the imported "
            "observer's forward pass"
        )

    def test_fig2_script_defines_no_observer_class_of_its_own_if_it_exists(self):
        script_path = _REPO_ROOT / "figures" / "fig2_ideal_observer.py"
        if not script_path.exists():
            pytest.skip("figures/fig2_ideal_observer.py has not been written yet")
        tree = ast.parse(script_path.read_text())
        observer_classes = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and "Observer" in node.name
        ]
        assert not observer_classes, (
            f"figures/fig2_ideal_observer.py defines its own observer-like "
            f"class(es): {observer_classes}"
        )


# --------------------------------------------------------------------------- #
# Output schema and behavioral contracts — need the real cached artifacts
# under data/cache/model_states/control_dataset/.
# --------------------------------------------------------------------------- #
_needs_real_data = pytest.mark.skipif(
    not _HAS_REAL_DATA, reason=f"real control_dataset artifacts not found under {_DATASET_DIR}"
)


@_needs_real_data
class TestOutputSchema:
    def test_hazard_sweep_has_exactly_the_documented_columns(self):
        df = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=_HAZARD_LEVELS,
            idx_time=2,
        )
        assert set(df.columns) == _EXPECTED_COLUMNS

    def test_contingency_sweep_has_exactly_the_documented_columns(self):
        df = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Bounce",
            sweep="contingency",
            levels=_CONTINGENCY_LEVELS,
        )
        assert set(df.columns) == _EXPECTED_COLUMNS

    def test_p_change_is_a_probability(self):
        df = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=_HAZARD_LEVELS,
            idx_time=2,
        )
        assert (df["p_change"] >= 0).all()
        assert (df["p_change"] <= 1).all()

    def test_boolean_columns_are_actually_boolean(self):
        df = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=_HAZARD_LEVELS,
            idx_time=2,
        )
        assert df["occluded"].dtype == bool
        assert df["is_bounce"].dtype == bool

    def test_level_is_an_ordered_categorical_from_low_to_high(self):
        df_hazard = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=_HAZARD_LEVELS,
            idx_time=2,
        )
        assert isinstance(df_hazard["level"].dtype, pd.CategoricalDtype)
        assert df_hazard["level"].dtype.ordered
        assert list(df_hazard["level"].dtype.categories) == ["Low", "High"]

        df_contingency = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Bounce",
            sweep="contingency",
            levels=_CONTINGENCY_LEVELS,
        )
        assert isinstance(df_contingency["level"].dtype, pd.CategoricalDtype)
        assert df_contingency["level"].dtype.ordered
        assert list(df_contingency["level"].dtype.categories) == ["Low", "Medium", "High"]

    def test_video_id_is_constant_within_one_call(self):
        df = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=_HAZARD_LEVELS,
            idx_time=2,
        )
        assert df["video_id"].nunique() == 1


@_needs_real_data
class TestObserverFingerprintParticipatesInTheCacheKey:
    """The fingerprint must actually change the cache key, not just exist.

    ``check_call_in_cache`` inspects the function's own source plus the
    hashed, ``filter_args``-normalized bound arguments without calling the
    function or touching the filesystem cache's data, so it is the correct
    joblib API for asserting membership: it answers exactly "is this call
    already cached" without the side effects (or cost) of ``call``.
    """

    def test_default_call_is_cached_but_an_altered_fingerprint_is_not(self):
        # Populate the cache for the default-argument call. This must not
        # use LIC_FIG_FORCE_RECOMPUTE, which wipes the shared cache rather
        # than scoping to this one call.
        transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=_HAZARD_LEVELS,
            idx_time=2,
        )
        assert transforms.ideal_observer_belief_curves.check_call_in_cache(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=_HAZARD_LEVELS,
            idx_time=2,
        ), (
            "the call just made is not reported as cached; "
            "check_call_in_cache disagrees with the call that just ran"
        )
        assert not transforms.ideal_observer_belief_curves.check_call_in_cache(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=_HAZARD_LEVELS,
            idx_time=2,
            observer_fingerprint="0" * 16,
        ), (
            "an explicit observer_fingerprint different from the real "
            "default is reported as already cached; the fingerprint is not "
            "participating in the cache key"
        )


@_needs_real_data
class TestBeliefCurveSemantics:
    def test_baseline_at_the_last_visible_frame_equals_the_swept_hazard(self):
        """On the last visible frame before the final occlusion, the belief
        has just been reset to the observed color and taken one propagation
        step, so the change probability equals the swept parameter exactly.
        """
        df = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=_HAZARD_LEVELS,
            idx_time=2,
            lead_in_frames=_LEAD_IN_FRAMES,
        )
        for name, param in _HAZARD_LEVELS:
            row = df[(df["level"] == name) & (df["frame"] == _LAST_VISIBLE_FRAME)]
            assert len(row) == 1, f"no row for level={name!r} at frame={_LAST_VISIBLE_FRAME}"
            assert row["p_change"].iloc[0] == pytest.approx(param, abs=1e-6)

    def test_p_change_never_exceeds_the_uniform_asymptote_after_the_anchor_frame(self):
        """With three colors and no re-anchoring, occluded belief cannot push
        the change probability above 2/3 (uniform over the two other colors).

        Scoped to the anchor frame onward. Earlier lead-in frames are visible
        frames like any other: the belief is re-anchored to whatever color is
        showing then, which need not be the color the curve is anchored to, and
        ``1 - belief[anchor]`` on such a frame is near 1 by construction rather
        than being a belief propagated from the anchor. The 2/3 asymptote is a
        property of propagation from the anchor, so that is where it is
        asserted.
        """
        df = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=_HAZARD_LEVELS,
            idx_time=2,
            lead_in_frames=_LEAD_IN_FRAMES,
        )
        anchored = df[df["frame"] >= _LAST_VISIBLE_FRAME]
        assert not anchored.empty
        assert (anchored["p_change"] <= 2 / 3 + 1e-6).all()

    def test_p_change_is_nondecreasing_across_occluded_frames_of_a_straight_trial(self):
        df = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=_HAZARD_LEVELS,
            idx_time=2,
        )
        for name, _param in _HAZARD_LEVELS:
            occluded = df[(df["level"] == name) & df["occluded"]].sort_values("frame")
            diffs = occluded["p_change"].diff().dropna()
            assert (diffs >= -1e-9).all(), f"{name} is not monotone under occlusion"

    def test_higher_hazard_is_pointwise_at_least_as_large_and_strictly_larger_at_the_end(self):
        """From the anchor frame onward, a higher hazard rate can only make a
        change more likely — including at the anchor frame itself, where the
        gap between the levels is exactly the difference in hazard rate.

        Not asserted over the earlier lead-in frames: on a visible frame
        showing the color that cyclically precedes the anchor color, the
        propagated belief in the anchor color *is* the hazard rate, so
        ``1 - belief[anchor]`` there is ``1 - hazard`` and the ordering
        structurally inverts. That inversion says nothing about the panel's
        claim, which is about belief propagated from the anchor.
        """
        df = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=_HAZARD_LEVELS,
            idx_time=2,
            lead_in_frames=_LEAD_IN_FRAMES,
        )
        pivot = df.pivot(index="frame", columns="level", values="p_change").sort_index()
        anchored = pivot.loc[_LAST_VISIBLE_FRAME:]
        assert not anchored.empty
        assert (anchored["High"] >= anchored["Low"] - 1e-9).all()
        assert anchored["High"].iloc[-1] > anchored["Low"].iloc[-1] + 1e-3

    def test_contingency_levels_coincide_before_the_bounce(self):
        """The task has no field controlling velocity-change probability
        independent of a wall bounce, so away from the bounce frame the
        change probability tracks the fixed hazard parameter only — every
        contingency level must therefore produce identical values before
        the shared bounce frame is reached.
        """
        df = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Bounce",
            sweep="contingency",
            levels=_CONTINGENCY_LEVELS,
            pvc=0.0,
        )
        bounce_frames = df.loc[df["is_bounce"], "frame"].unique()
        assert len(bounce_frames) == 1, "the bounce frame must be shared across contingency levels"
        bounce_frame = bounce_frames[0]

        pre_bounce = df[df["frame"] < bounce_frame]
        pivot = pre_bounce.pivot(index="frame", columns="level", values="p_change").sort_index()
        reference = pivot.iloc[:, 0]
        for col in pivot.columns[1:]:
            np.testing.assert_allclose(
                pivot[col].to_numpy(), reference.to_numpy(), atol=1e-9,
                err_msg=f"level {col!r} diverges from {pivot.columns[0]!r} before the bounce",
            )

    def test_contingency_levels_separate_and_order_high_over_medium_over_low_after_the_bounce(self):
        df = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Bounce",
            sweep="contingency",
            levels=_CONTINGENCY_LEVELS,
            pvc=0.0,
        )
        end_frame = df["frame"].max()
        pivot = df.pivot(index="frame", columns="level", values="p_change")
        end = pivot.loc[end_frame]
        assert end["High"] >= end["Medium"] >= end["Low"]
        assert end["High"] > end["Low"] + 0.05

    def test_the_largest_step_lands_on_the_frame_flagged_as_a_bounce(self):
        """The bounce marker and the step it explains must be the same frame.

        Nothing else in this file ties the ``is_bounce`` column to the curve:
        an off-by-one in the marker, or a marker derived from a different
        criterion than the one the observer's own transition math uses, would
        leave every other test green while drawing the vertical line beside the
        step instead of on it.
        """
        df = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Bounce",
            sweep="contingency",
            levels=_CONTINGENCY_LEVELS,
            pvc=0.0,
        )
        bounce_frames = df.loc[df["is_bounce"], "frame"].unique()
        assert len(bounce_frames) == 1
        bounce_frame = bounce_frames[0]

        for name, _param in _CONTINGENCY_LEVELS:
            curve = df[df["level"] == name].sort_values("frame")
            steps = curve.set_index("frame")["p_change"].diff().dropna()
            assert steps.idxmax() == bounce_frame, (
                f"level {name!r}: the largest step is at frame "
                f"{steps.idxmax()}, not at the flagged bounce frame "
                f"{bounce_frame}"
            )

    def test_repeated_calls_are_deterministic(self):
        first = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=_HAZARD_LEVELS,
            idx_time=2,
        )
        second = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=_HAZARD_LEVELS,
            idx_time=2,
        )
        pd.testing.assert_frame_equal(first, second)

    def test_changed_levels_yield_a_different_frame(self):
        default = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=_HAZARD_LEVELS,
            idx_time=2,
        )
        other = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=(("High", 0.05), ("Low", 0.01)),
            idx_time=2,
        )
        assert not default["p_change"].equals(other["p_change"])


# --------------------------------------------------------------------------- #
# Oracle tier — checks the transform's output against a documented parameter
# regime for the task. The bands below are intentionally wide enough to
# admit either a nominal or an estimated reading of the task's hazard and
# contingency parameters for this exemplar, rather than pinning one exact
# numeric reading: both are defensible characterizations of the same
# quantity, and the ordering/separation the observer's math guarantees is
# the property under test, not a single-decimal match.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.integration
@_needs_real_data
class TestBeliefCurveAgreesWithDocumentedParameterRegime:
    def test_hazard_terminal_values_fall_within_the_documented_band(self):
        df = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Straight",
            sweep="hazard",
            levels=_HAZARD_LEVELS,
            idx_time=2,
        )
        end_frame = df["frame"].max()
        pivot = df.pivot(index="frame", columns="level", values="p_change")
        high_end = pivot.loc[end_frame, "High"]
        low_end = pivot.loc[end_frame, "Low"]

        assert 0.45 <= high_end <= 0.60
        assert 0.20 <= low_end <= 0.32
        assert high_end - low_end >= 0.15

    def test_contingency_plateau_ratios_fall_within_the_documented_band(self):
        df = transforms.ideal_observer_belief_curves(
            dataset=_DATASET,
            trial_type="Bounce",
            sweep="contingency",
            levels=_CONTINGENCY_LEVELS,
            pvc=0.0,
        )
        end_frame = df["frame"].max()
        pivot = df.pivot(index="frame", columns="level", values="p_change")
        end = pivot.loc[end_frame]

        assert end["High"] > end["Medium"] > end["Low"]
        ratio_medium = end["Medium"] / end["High"]
        ratio_low = end["Low"] / end["High"]
        assert 0.45 <= ratio_medium <= 0.80
        assert 0.10 <= ratio_low <= 0.45

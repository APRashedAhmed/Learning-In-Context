"""Contract tests for the NOT-YET-WRITTEN
``src/learning_in_context/visualization/transforms.py`` (figures/SPEC.md rule 8,
procedure step 2).

This module does not exist yet, so the top-level import below is expected to
raise ``ModuleNotFoundError`` and every test here should be RED until the
module is implemented.

Cache-dir seam (judgment call, encoded here as the contract):
    ``transforms.get_memory(cache_dir: str | Path | None = None) -> joblib.Memory``
    is a factory. With ``cache_dir=None`` it reads the env var
    ``LIC_FIG_CACHE_DIR``; if that is unset too, it defaults to
    ``<repo_root>/data/cache/fig_transforms`` (SPEC rule 8's stated location,
    "cache lives beside tier-1 artifacts in data/cache/", per the 2026-08-28
    operator ruling in SPEC). ``transforms.MEMORY`` is the shared instance
    built by calling ``get_memory()`` at import time — this is what fig5's
    (and later fig6/fig4's) memoized transforms are decorated with.

    Tests use the explicit ``cache_dir=`` override to get an isolated
    ``Memory`` per test (never touching the real ``data/cache/fig_transforms``
    dir); one test also exercises the env-var path via monkeypatch to prove
    both seams work, since the SPEC left the choice open ("pick one and
    encode it").

FORCE_RECOMPUTE seam (judgment call): ``transforms.FORCE_RECOMPUTE`` is a
bool computed from the env var ``LIC_FIG_FORCE_RECOMPUTE`` at import time,
and ``transforms.clear_cache(memory=None)`` is the manual-recompute knob
SPEC rule 8 requires ("manual recompute = memory.clear(), delete the cache
dir, or a FORCE_RECOMPUTE env knob") — it clears the given Memory instance
(defaulting to the shared ``transforms.MEMORY``). This test module checks
``clear_cache`` behaviorally (recompute happens after clearing) and checks
``FORCE_RECOMPUTE`` only for existence/type, since wiring it into "clear
before every call" is an implementation detail left to the figure scripts.

Shared transform naming (judgment call — THE CONTRACT the implementer must
satisfy): SPEC rule 8 names fig5's shared transform singularly — "the
per-model intervention frames [...] and fig5's ordered-change windows (feed
both activity panels and profile scatters)". Reading
``figures/fig_hazard_rate_activity.py`` (the retrofit source, cells around
:334-:418 and :876-:999) shows this is actually two distinct computations,
both windowed around a color change but otherwise independent — there is no
code path where one literally feeds the other:

  * ``get_ordered_sliding_window`` (:334) builds the per-timestep, per-unit,
    per-change-order dataframe consumed by the activity time-course panels
    (``plot_ordered_change_activity_rows``, :674). This test module names the
    memoized equivalent ``ordered_change_windows`` and treats it as THE
    representative shared transform (deep signature contract below).
  * ``get_activity_difference_around_criterion`` /
    ``get_activity_difference_during_zero_criterion`` (:876, :908) build the
    step-size / activity-decay diffs consumed by the profile scatter panels
    (``plot_decay_vs_step``, :1024). This test module names the memoized
    equivalent ``activity_change_profile`` and only checks it exists, is
    callable, and avoids raw-array parameters (lighter contract).

A verifier should confirm this two-function split (vs. a single unified
"ordered-change windows" function) matches operator intent — it is the
most defensible reading of the source, but SPEC's wording suggests the
author may have had one function in mind.

Path/param-keying contract (SPEC rule 8: "Key on paths + params, never on
loaded arrays"): the notebook's originals take pre-loaded numpy arrays
(``states``, ``samples``, ``targets``, ``df_selected``) as arguments — that
is exactly the anti-pattern rule 8 forbids for a memoized function (it would
force joblib to hash large arrays on every call, and the cache key would be
unstable/expensive). The memoized ``transforms.py`` versions must instead key
on identifiers (dataset name, model name, exp id, ...) and load data
internally. This is checked via ``inspect.signature`` heuristics rather than
pinning an exact parameter list, so the implementer has latitude in naming.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import joblib
import pytest

# Import defensively: transforms.py does not exist yet. A bare top-level
# import would raise a *collection* error that aborts pytest's entire run
# (verified empirically: a single module's ImportError at collection time
# stops ALL other test modules in the same invocation, not just this one) --
# that would break test_paper_style.py's green run too. Guarding the import
# and letting each test fail independently on ``transforms.<attr>`` (raising
# AttributeError on the ``None`` sentinel) gives per-test RED failures for
# the right reason instead of one blanket collection abort.
try:
    from learning_in_context.visualization import transforms
    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised until transforms.py exists
    transforms = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_transforms_module_is_importable():
    """Standalone, clearly-named red flag for "transforms.py doesn't exist yet".

    Every other test below will also fail (via AttributeError on the ``None``
    sentinel) if this one does, but this test names the actual root cause.
    """
    if _IMPORT_ERROR is not None:
        pytest.fail(
            "learning_in_context.visualization.transforms is not importable "
            f"yet: {_IMPORT_ERROR!r}"
        )

# Parameter names that would indicate a memoized function is keying on
# preloaded arrays rather than paths/params (SPEC rule 8).
_FORBIDDEN_ARRAY_PARAM_NAMES = {
    "states",
    "samples",
    "targets",
    "df_selected",
    "df_data",
    "array",
    "arrays",
    "data",
}


class TestMemoryInstance:
    """A shared joblib.Memory instance, rooted at data/cache/fig_transforms."""

    def test_module_exposes_memory_instance(self):
        assert isinstance(transforms.MEMORY, joblib.Memory)

    def test_module_exposes_get_memory_factory(self):
        assert callable(transforms.get_memory)

    def test_default_root_is_data_cache_fig_transforms(self, monkeypatch):
        monkeypatch.delenv("LIC_FIG_CACHE_DIR", raising=False)
        memory = transforms.get_memory()
        expected = _REPO_ROOT / "data" / "cache" / "fig_transforms"
        assert Path(memory.location) == expected

    def test_cache_dir_overridable_via_explicit_factory_arg(self, tmp_path):
        memory = transforms.get_memory(cache_dir=tmp_path / "custom_cache")
        assert Path(memory.location) == tmp_path / "custom_cache"

    def test_cache_dir_overridable_via_env_var(self, tmp_path, monkeypatch):
        override = tmp_path / "env_cache"
        monkeypatch.setenv("LIC_FIG_CACHE_DIR", str(override))
        memory = transforms.get_memory()
        assert Path(memory.location) == override


class TestForceRecompute:
    """Manual recompute: FORCE_RECOMPUTE env knob and/or memory.clear() passthrough."""

    def test_force_recompute_flag_exists_and_is_bool(self):
        assert isinstance(transforms.FORCE_RECOMPUTE, bool)

    def test_clear_cache_is_callable(self):
        assert callable(transforms.clear_cache)

    def test_clear_cache_forces_recompute(self, tmp_path):
        memory = transforms.get_memory(cache_dir=tmp_path / "cache")
        calls = {"n": 0}

        @memory.cache
        def _dummy(x):
            calls["n"] += 1
            return x * 2

        assert _dummy(3) == 6
        assert _dummy(3) == 6
        assert calls["n"] == 1  # second call hit the cache

        transforms.clear_cache(memory=memory)

        assert _dummy(3) == 6
        assert calls["n"] == 2  # recomputed after clear_cache


class TestMemoizationInfrastructure:
    """Infrastructure-level check: functions decorated with the shared Memory
    actually memoize (call-twice-hits-cache), using the cache-dir seam so no
    real cache directory is touched.
    """

    def test_call_twice_hits_cache(self, tmp_path):
        memory = transforms.get_memory(cache_dir=tmp_path / "cache")
        calls = {"n": 0}

        @memory.cache
        def _dummy_transform(dataset: str, model_name: str, param: int):
            calls["n"] += 1
            return {"dataset": dataset, "model_name": model_name, "param": param}

        first = _dummy_transform("extended_dataset", "lstm", 16)
        second = _dummy_transform("extended_dataset", "lstm", 16)

        assert first == second
        assert calls["n"] == 1

    def test_different_params_do_not_share_cache_entry(self, tmp_path):
        memory = transforms.get_memory(cache_dir=tmp_path / "cache")
        calls = {"n": 0}

        @memory.cache
        def _dummy_transform(dataset: str, T: int):
            calls["n"] += 1
            return T

        _dummy_transform("extended_dataset", 16)
        _dummy_transform("extended_dataset", 32)

        assert calls["n"] == 2


class TestOrderedChangeWindowsSignature:
    """The representative shared transform (SPEC rule 8's callout).

    Feeds the fig5 activity time-course panels. Must key on paths/ids/params,
    never on preloaded arrays.
    """

    def test_exists_and_is_callable(self):
        assert hasattr(transforms, "ordered_change_windows")
        assert callable(transforms.ordered_change_windows)

    def test_no_raw_array_parameters(self):
        sig = inspect.signature(transforms.ordered_change_windows)
        param_names = set(sig.parameters.keys())
        offending = param_names & _FORBIDDEN_ARRAY_PARAM_NAMES
        assert not offending, (
            f"ordered_change_windows takes raw-array-shaped params {offending}; "
            "SPEC rule 8 requires keying on paths/ids/params, loading data "
            "internally."
        )

    def test_no_array_typed_defaults(self):
        # A default value that is itself a large object (e.g. a numpy array
        # or DataFrame) would defeat the point of keying on paths/params.
        sig = inspect.signature(transforms.ordered_change_windows)
        for param in sig.parameters.values():
            if param.default is inspect.Parameter.empty:
                continue
            assert isinstance(
                param.default, (str, int, float, bool, type(None), Path, tuple)
            ), (
                f"ordered_change_windows param {param.name!r} has a non-primitive "
                f"default ({type(param.default)!r}); expected paths/ids/params only "
                "(a small hashable tuple such as a window/order spec is allowed)."
            )

    def test_takes_an_identifier_parameter(self):
        # At least one parameter must look like a dataset/model/experiment
        # identifier rather than a bare numeric knob — otherwise the function
        # has no path/id to key the cache on at all.
        sig = inspect.signature(transforms.ordered_change_windows)
        param_names = set(sig.parameters.keys())
        identifier_like = {
            name
            for name in param_names
            if any(token in name for token in ("dataset", "model", "exp_id", "path"))
        }
        assert identifier_like, (
            "ordered_change_windows has no dataset/model/exp_id/path-like "
            f"parameter among {sorted(param_names)}"
        )

    def test_is_memoized_with_the_shared_memory_instance(self):
        # joblib.Memory.cache wraps the function in a MemorizedFunc whose
        # .func attribute is the original callable, and whose store_backend
        # is tied to the Memory instance it was decorated with.
        fn = transforms.ordered_change_windows
        assert hasattr(fn, "func") and hasattr(fn, "store_backend"), (
            "ordered_change_windows does not look like a joblib-memoized function"
        )
        # SPEC rule 8: decorated with the *shared* Memory instance, not a
        # private/local Memory. joblib roots a MemorizedFunc's store under
        # ``<memory.location>/joblib`` — so the function's backend location
        # must sit under transforms.MEMORY's location.
        mem_root = Path(transforms.MEMORY.location).resolve()
        fn_loc = Path(fn.store_backend.location).resolve()
        assert fn_loc == mem_root or mem_root in fn_loc.parents, (
            "ordered_change_windows is memoized with a Memory rooted at "
            f"{fn_loc}, not the shared transforms.MEMORY at {mem_root}"
        )


class TestActivityChangeProfileExists:
    """Lighter contract for the profile-scatter-feeding transform (see module
    docstring for why this is a second, separately-named function rather than
    a reuse of ordered_change_windows).
    """

    def test_exists_and_is_callable(self):
        assert hasattr(transforms, "activity_change_profile")
        assert callable(transforms.activity_change_profile)

    def test_no_raw_array_parameters(self):
        sig = inspect.signature(transforms.activity_change_profile)
        param_names = set(sig.parameters.keys())
        offending = param_names & _FORBIDDEN_ARRAY_PARAM_NAMES
        assert not offending, (
            f"activity_change_profile takes raw-array-shaped params {offending}"
        )

    def test_is_memoized_with_the_shared_memory_instance(self):
        fn = transforms.activity_change_profile
        assert hasattr(fn, "func") and hasattr(fn, "store_backend"), (
            "activity_change_profile does not look like a joblib-memoized function"
        )
        mem_root = Path(transforms.MEMORY.location).resolve()
        fn_loc = Path(fn.store_backend.location).resolve()
        assert fn_loc == mem_root or mem_root in fn_loc.parents, (
            "activity_change_profile is memoized with a Memory rooted at "
            f"{fn_loc}, not the shared transforms.MEMORY at {mem_root}"
        )

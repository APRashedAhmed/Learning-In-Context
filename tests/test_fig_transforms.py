"""Contract tests for ``src/learning_in_context/visualization/transforms.py``.

Cache-dir seam:
    ``transforms.get_memory(cache_dir: str | Path | None = None) -> joblib.Memory``
    is a factory. With ``cache_dir=None`` it reads the env var
    ``LIC_FIG_CACHE_DIR``; if that is unset too, it defaults to
    ``<repo_root>/data/cache/fig_transforms`` — the transform cache lives
    beside the tier-1 artifacts under ``data/cache/``. ``transforms.MEMORY``
    is the shared instance built by calling ``get_memory()`` at import time,
    and every memoized figure transform is decorated with it.

    Tests use the explicit ``cache_dir=`` override to get an isolated
    ``Memory`` per test (never touching the real ``data/cache/fig_transforms``
    dir); one test also exercises the env-var path via monkeypatch, so both
    seams are covered.

FORCE_RECOMPUTE seam: ``transforms.FORCE_RECOMPUTE`` is a bool computed from
the env var ``LIC_FIG_FORCE_RECOMPUTE`` at import time, and
``transforms.clear_cache(memory=None)`` is the manual-recompute knob — it
clears the given Memory instance (defaulting to the shared
``transforms.MEMORY``). This module checks ``clear_cache`` behaviorally
(recompute happens after clearing) and checks ``FORCE_RECOMPUTE`` only for
existence/type, since wiring it into "clear before every call" is left to the
figure scripts.

Shared transform naming: fig5's shared work is two distinct computations, both
windowed around a color change but otherwise independent — there is no code
path where one feeds the other — so ``transforms.py`` exposes two functions
rather than one:

  * ``ordered_change_windows`` — the per-timestep, per-unit, per-change-order
    dataframe consumed by the activity time-course panels. Treated here as the
    representative shared transform (deep signature contract below).
  * ``activity_change_profile`` — the step-size / activity-decay diffs consumed
    by the profile scatter panels. Lighter contract: exists, is callable, and
    avoids raw-array parameters.

Path/param-keying contract: the source notebooks' originals take pre-loaded
numpy arrays (``states``, ``samples``, ``targets``, ``df_selected``) as
arguments, which is exactly the anti-pattern for a memoized function — it would
force joblib to hash large arrays on every call, making the cache key unstable
and expensive. The memoized ``transforms.py`` versions key on identifiers
(dataset name, model name, exp id, ...) and load data internally. This is
checked via ``inspect.signature`` heuristics rather than by pinning an exact
parameter list, leaving latitude in naming.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import joblib
import pytest

# Import defensively. If transforms.py were unimportable, a bare top-level
# import would raise a *collection* error that aborts pytest's entire run:
# a single module's ImportError at collection time stops ALL other test
# modules in the same invocation, not just this one. Guarding the import and
# letting each test fail independently on ``transforms.<attr>`` (raising
# AttributeError on the ``None`` sentinel) gives per-test failures for the
# right reason instead of one blanket collection abort.
try:
    from learning_in_context.visualization import transforms
    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - only when transforms.py is broken
    transforms = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_transforms_module_is_importable():
    """Standalone, clearly-named check that transforms.py imports at all.

    Every other test below will also fail (via AttributeError on the ``None``
    sentinel) if this one does, but this test names the actual root cause.
    """
    if _IMPORT_ERROR is not None:
        pytest.fail(
            "learning_in_context.visualization.transforms is not importable: "
            f"{_IMPORT_ERROR!r}"
        )

# Parameter names that would indicate a memoized function is keying on
# preloaded arrays rather than paths/params.
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
    """The representative shared transform.

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
            "memoized transforms must key on paths/ids/params and load data "
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
        # Must be decorated with the *shared* Memory instance, not a
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

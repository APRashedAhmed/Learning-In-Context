"""Shared, memoized figure transformations (figures/SPEC.md rule 8).

Tier-2 of the three-tier figure pipeline: everything between the tier-1 cached
model-state artifacts (``data/cache/model_states/``) and a plot-ready DataFrame.
Transforms are pure functions decorated with a single shared ``joblib.Memory``
instance so results are cached once and reused across every figure script.

Cache-dir seam
--------------
:func:`get_memory` is a factory. With ``cache_dir=None`` it reads the env var
``LIC_FIG_CACHE_DIR``; if that is unset it defaults to
``<repo_root>/data/cache/fig_transforms`` (SPEC rule 8: "cache lives beside
tier-1 artifacts in data/cache/"). The module-level :data:`MEMORY` is the shared
instance every memoized transform below is decorated with. Tests pass an
explicit ``cache_dir=`` (or set the env var) to isolate a throwaway cache.

Keying discipline (SPEC rule 8)
-------------------------------
Every memoized transform keys on **paths / ids / params only** (dataset name,
model name, experiment id, small numeric knobs, hashable tuples) and loads the
underlying arrays internally — never accepting a preloaded ``states`` /
``samples`` / ``targets`` array as an argument. That keeps the cache key small
and stable instead of forcing joblib to hash multi-GB arrays on every call.

Two transforms are exposed (the two-function split is the test contract — the
tests import both by name):

* :func:`ordered_change_windows` — the per-timestep, per-unit, per-change-order
  windows that feed fig5's activity time-course panels.
* :func:`activity_change_profile` — the per-model step-size / activity-decay
  table that feeds fig5's activity-profile scatter panels.

Both are ported (internals verbatim) from ``figures/fig_hazard_rate_activity.py``
and ``figures/fig_contingency_activity.py``; the sibling scripts are left
untouched.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided, sliding_window_view
from scipy import stats

# Repo root: this file lives at src/learning_in_context/visualization/.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "cache" / "fig_transforms"
_MODEL_STATES_ROOT = _REPO_ROOT / "data" / "cache" / "model_states"


# --------------------------------------------------------------------------- #
# Memory infrastructure
# --------------------------------------------------------------------------- #
def get_memory(cache_dir: str | Path | None = None) -> joblib.Memory:
    """Build a ``joblib.Memory`` rooted at the fig-transform cache directory.

    Resolution order (evaluated at call time):
        1. explicit ``cache_dir`` argument,
        2. the ``LIC_FIG_CACHE_DIR`` environment variable,
        3. ``<repo_root>/data/cache/fig_transforms`` (SPEC rule 8).
    """
    if cache_dir is None:
        env = os.environ.get("LIC_FIG_CACHE_DIR")
        cache_dir = env if env else _DEFAULT_CACHE_DIR
    return joblib.Memory(location=str(cache_dir), verbose=0)


# Shared instance every memoized transform below is decorated with.
MEMORY = get_memory()

# Manual-recompute knob (SPEC rule 8): FORCE_RECOMPUTE reflects the env var at
# import time; figure scripts may consult it to clear before a run.
FORCE_RECOMPUTE = os.environ.get("LIC_FIG_FORCE_RECOMPUTE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def clear_cache(memory: joblib.Memory | None = None) -> None:
    """Drop all memoized results (SPEC rule 8's ``memory.clear()`` knob).

    Clears the given ``Memory`` instance, defaulting to the shared
    :data:`MEMORY`.
    """
    (memory or MEMORY).clear(warn=False)


# --------------------------------------------------------------------------- #
# Ported constants (exemplar critical units per model)
# --------------------------------------------------------------------------- #
# Hidden-unit index (< hidden_size) is listed FIRST in each model's dict so the
# insertion order is (hidden, cell); downstream code relies on that ordering.
# Ported from figures/fig_hazard_rate_activity.py:790 ("hz", 10 models) and
# figures/fig_contingency_activity.py:1089 ("cont", 6 models).
STAT_UNITS: dict[str, dict[str, dict[int, str]]] = {
    "hz": {
        "san-4602": {1: "Unit 1 - Hidden 1", 17: "Unit 17 - Cell 1"},
        "san-4605": {8: "Unit 8 - Hidden 8", 24: "Unit 24 - Cell 8"},
        "san-4604": {15: "Unit 15 - Hidden 15", 31: "Unit 31 - Cell 15"},
        "san-4603": {6: "Unit 6 - Hidden 6", 22: "Unit 22 - Cell 6"},
        "san-4606": {11: "Unit 11 - Hidden 11", 27: "Unit 27 - Cell 11"},
        "san-4601": {1: "Unit 1 - Hidden 1", 17: "Unit 17 - Cell 1"},
        "san-4615": {4: "Unit 4 - Hidden 4", 20: "Unit 20 - Cell 4"},
        "san-4616": {5: "Unit 5 - Hidden 5", 21: "Unit 21 - Cell 5"},
        "san-4618": {4: "Unit 4 - Hidden 4", 20: "Unit 20 - Cell 4"},
        "san-4617": {0: "Unit 0 - Hidden 0", 16: "Unit 16 - Cell 0"},
    },
    "cont": {
        "san-4604": {11: "Unit 11 - Hidden 11", 27: "Unit 27 - Cell 11"},
        "san-4603": {0: "Unit 0 - Hidden 0", 16: "Unit 16 - Cell 0"},
        "san-4606": {12: "Unit 12 - Hidden 12", 28: "Unit 28 - Cell 12"},
        "san-4601": {5: "Unit 5 - Hidden 5", 21: "Unit 21 - Cell 5"},
        "san-4615": {12: "Unit 12 - Hidden 12", 28: "Unit 28 - Cell 12"},
        "san-4618": {9: "Unit 9 - Hidden 9", 25: "Unit 25 - Cell 9"},
    },
}

# Models with no contingency manipulation — excluded from the contingency
# blocks (fig_contingency_activity.py:182).
EXP_NO_CONT = frozenset({"san-4602", "san-4605", "san-4616", "san-4617"})

# Contingency-block PROFILE unit set (deck-verified, see fig5 docstring): the
# "hz" exemplar units restricted to the 6 contingency models. Numerically
# confirmed against the deck page ("Fig 5 - Crit Unit Behavior-1.png"): the
# deck's contingency profile decays/steps match these units exactly (decay is
# criterion-independent, so matching decays fingerprint the unit set), while
# the "cont" exemplar units do not.
STAT_UNITS["hz_cont"] = {
    exp_id: unit_map
    for exp_id, unit_map in STAT_UNITS["hz"].items()
    if exp_id not in EXP_NO_CONT
}

DEFAULT_DATASET = "extended_dataset"
DEFAULT_MODEL = "lstm"


# --------------------------------------------------------------------------- #
# Internal data loaders (never exposed as memoized-function arguments)
# --------------------------------------------------------------------------- #
def _dataset_dir(dataset: str) -> Path:
    return _MODEL_STATES_ROOT / dataset


def _load_dataset(dataset: str):
    """Load samples, targets (padding-masked), and trial metadata for a dataset.

    Ported verbatim from the ``## Loading Data`` cell of the source notebooks.
    """
    base = _dataset_dir(dataset)
    samples = np.load(str(base / "samples.npy"), allow_pickle=True)
    targets = np.load(str(base / "targets.npy"), allow_pickle=True)
    batch_size, timesteps, _ = samples.shape

    timestep_array = np.tile(np.arange(timesteps), batch_size).reshape(
        batch_size, timesteps
    )
    df_data = pd.read_csv(base / "trial_meta.csv", index_col=0)
    with open(str(base / "dataset_meta.pkl"), "rb") as f:
        dict_metadata = pickle.load(f)

    padding_value = dict_metadata["padding_value"]
    length = df_data["length"].values
    mask_valid = (timestep_array < length[:, None])[:, :, None]
    targets = np.where(mask_valid, targets, padding_value)
    return samples, targets, df_data, padding_value


def _load_first_states_zscore(dataset: str, model_name: str, exp_id: str, M: int):
    """z-scored concat of the first ``M`` hidden+cell states for one model.

    Ported from the ``dict_model_first_states`` construction in the sources.
    """
    path = _dataset_dir(dataset) / model_name / f"{exp_id}.npz"
    model_data = np.load(str(path), allow_pickle=True)
    return stats.zscore(
        np.concatenate(
            [model_data["hiddens"][:, :M], model_data["cells"][:, :M]],
            axis=-1,
        ),
        axis=(0, 1),
        ddof=1,
    )


def _load_raw_states(dataset: str, model_name: str, exp_id: str):
    """The raw ``states`` array for one model (contingency time-courses)."""
    path = _dataset_dir(dataset) / model_name / f"{exp_id}.npz"
    model_data = np.load(str(path), allow_pickle=True)
    return model_data["states"]


# --------------------------------------------------------------------------- #
# Windowing / diff kernels (ported internals)
# --------------------------------------------------------------------------- #
def _ordered_sliding_window(states, samples, targets, df_selected, T, k, change_idx, mask_mode):
    """Center T-length windows on the k-th change target at ``change_idx``.

    Ported from ``get_ordered_sliding_window`` (single_k) and
    ``get_ordered_sliding_window_bounce_no_change`` (bounce_no_change).
    """
    states = states[df_selected.index]
    samples = samples[df_selected.index]
    targets = targets[df_selected.index]

    states_win = sliding_window_view(states, window_shape=T, axis=1)
    sample_win = sliding_window_view(samples, window_shape=T, axis=1)
    target_win = sliding_window_view(targets, window_shape=T, axis=1)

    if mask_mode == "bounce_no_change":
        mask_center = (target_win[:, :, -4, change_idx] == 1) & (
            target_win[:, :, -2, change_idx] == 0
        )
    else:  # "single_k"
        mask_center = target_win[:, :, -k, change_idx] == 1

    states_win = np.moveaxis(states_win[mask_center], [2], [1])
    sample_win = np.moveaxis(sample_win[mask_center], [2], [1])
    target_win = np.moveaxis(target_win[mask_center], [2], [1])

    order_counts = np.cumsum(mask_center, axis=1) - 1
    order_vector = order_counts[mask_center].tolist()

    return states_win, sample_win, target_win, order_vector


def _diff_around_criterion(states, targets, df_selected, tau, criterion_mode):
    """Signed state change ``tau`` steps before vs after a criterion event.

    Ported from ``get_activity_difference_around_criterion`` (the criterion mask
    varies by block).
    """
    states = states[df_selected.index]
    targets = targets[df_selected.index]

    num_trials, timesteps, num_features = states.shape

    if criterion_mode == "color_change":
        criterion_mask = targets[:, :, -1] == 1
    elif criterion_mode == "bounce_color_change":
        # Contingent (bounce-triggered) color change. NOTE: the literal source
        # cell (fig_contingency_activity.py:1170) masks on
        # ``(targets[:, :, -4:-2] == 1).any(-1)`` (every bounce), but that cell
        # is dead code (empty "hz" dict) and its criterion does NOT reproduce
        # the deck's block-2 profile panel; ``targets[:, :, -2] == 1`` does,
        # exactly (verified numerically against the deck page).
        criterion_mask = targets[:, :, -2] == 1
    elif criterion_mode == "bounce_no_change":
        criterion_mask = (targets[:, :, -4] == 1) & (targets[:, :, -2] == 0)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unknown criterion_mode {criterion_mode!r}")

    trial_indices, time_indices = np.where(criterion_mask)
    valid_mask = (time_indices >= tau) & (time_indices < timesteps - tau)
    trial_indices = trial_indices[valid_mask]
    time_indices = time_indices[valid_mask]

    states_before = states[trial_indices, time_indices - tau]
    states_after = states[trial_indices, time_indices + tau]
    return states_after - states_before


def _diff_during_zero_criterion(states, targets, df_selected, tau):
    """Mean per-step drift across ``tau`` consecutive no-change steps.

    Ported verbatim from ``get_activity_difference_during_zero_criterion``.
    """
    states = states[df_selected.index]
    targets = targets[df_selected.index]

    num_trials, timesteps, num_features = states.shape

    zero_mask = (targets[:, :, -4:] == 0).all(axis=-1)

    stride_time = zero_mask.strides[1]
    zero_windows = as_strided(
        zero_mask,
        shape=(num_trials, timesteps - tau + 1, tau),
        strides=(zero_mask.strides[0], stride_time, stride_time),
    )
    valid_sequences = zero_windows.all(axis=2)

    trial_indices, window_indices = np.where(valid_sequences)
    if len(trial_indices) == 0:
        return np.empty((0, num_features))

    flat_idx_start = trial_indices * timesteps + window_indices
    flat_idx_end = trial_indices * timesteps + (window_indices + tau - 1)

    states_flat = states.reshape(-1, num_features)
    states_start = states_flat[flat_idx_start]
    states_end = states_flat[flat_idx_end]
    return (states_end - states_start) / (tau - 1)


# --------------------------------------------------------------------------- #
# Memoized transforms (the shared tier-2 API)
# --------------------------------------------------------------------------- #
@MEMORY.cache
def ordered_change_windows(
    dataset: str,
    model_name: str,
    exp_id: str,
    split_col: str,
    units: tuple[int, ...],
    T: int = 16,
    k: int = 1,
    change_idx: int = 5,
    M: int = 250,
    state_source: str = "first_m_zscore",
    mask_mode: str = "single_k",
    order_prefix: str = "Change",
) -> pd.DataFrame:
    """Per-timestep, per-unit, per-change-order activity windows for one model.

    Feeds the fig5 activity time-course panels. Loads the model's states,
    samples, and targets internally (keyed only on ids/params per SPEC rule 8),
    windows them around the k-th change target at ``change_idx``, splits by
    ``split_col`` (e.g. "Hazard Rate" / "Contingency"), and returns a tidy long
    DataFrame with columns ``[condition, unit, order, Timestep, Value]``.

    Args:
        dataset: model-states dataset name (e.g. ``"extended_dataset"``).
        model_name: sub-directory under the dataset (e.g. ``"lstm"``).
        exp_id: exemplar model id (e.g. ``"san-4604"``).
        split_col: trial-metadata column to split conditions on.
        units: unit indices to extract (hidden first, then cell).
        state_source: ``"first_m_zscore"`` (hazard block — first ``M`` steps of
            z-scored concat(hiddens, cells), samples/targets sliced to ``M``) or
            ``"raw_states"`` (contingency blocks — raw ``states``, full arrays).
        mask_mode: ``"single_k"`` or ``"bounce_no_change"``.
        order_prefix: order-label prefix (``"Change"`` / ``"Color Change"`` /
            ``"No Color Change"``).
    """
    samples, targets, df_data, _padding = _load_dataset(dataset)

    if state_source == "raw_states":
        states = _load_raw_states(dataset, model_name, exp_id)
        samples_used = samples
        targets_used = targets
    else:  # "first_m_zscore"
        states = _load_first_states_zscore(dataset, model_name, exp_id, M)
        samples_used = samples[:, :M]
        targets_used = targets[:, :M]

    frames = []
    for cond, df_cond in df_data.groupby(split_col):
        states_win, _sample_win, _target_win, order = _ordered_sliding_window(
            states, samples_used, targets_used, df_cond, T, k, change_idx, mask_mode
        )

        unit_dfs = []
        for unit in units:
            unit_activity = states_win[:, :, unit]
            df_unit = pd.DataFrame(unit_activity)
            df_unit = df_unit.assign(
                unit=unit,
                order=[f"{order_prefix} {j + 1}" for j in order],
            )
            unit_dfs.append(df_unit)

        melted = pd.concat(unit_dfs).melt(
            id_vars=["unit", "order"],
            var_name="Timestep",
            value_name="Value",
        )
        melted = melted.assign(condition=cond)
        frames.append(melted)

    return pd.concat(frames, ignore_index=True)


@MEMORY.cache
def activity_change_profile(
    dataset: str,
    model_name: str,
    criterion_mode: str,
    unit_set: str,
    tau_change: int = 5,
    tau_no_change: int = 2,
    M: int = 250,
) -> pd.DataFrame:
    """Per-model step-size / activity-decay table for the profile scatters.

    Feeds the fig5 activity-profile scatter panels. For every model in
    ``STAT_UNITS[unit_set]`` it loads the z-scored first-``M`` states internally
    and computes, for that model's hidden and cell exemplar units, the mean
    absolute "step" (change around a criterion event) and "decay" (drift during
    consecutive no-change steps). Returns a DataFrame indexed by model id with
    columns ``hidden_step, hidden_decay, cell_step, cell_decay``.

    Args:
        criterion_mode: ``"color_change"`` (hazard block),
            ``"bounce_color_change"`` (contingency change block), or
            ``"bounce_no_change"`` (contingency no-change block).
        unit_set: which exemplar-unit mapping to use — ``"hz"`` (all 10
            models; hazard block), ``"hz_cont"`` (hz units of the 6
            contingency models; both contingency profile blocks,
            deck-verified), or ``"cont"`` (contingency exemplar units, used
            by the time-course panels' unit choice, kept for reference).
    """
    samples, targets, df_data, _padding = _load_dataset(dataset)
    targets_M = targets[:, :M]

    model_units = STAT_UNITS[unit_set]
    records: dict[str, dict] = {}
    for exp_id, unit_map in model_units.items():
        states = _load_first_states_zscore(dataset, model_name, exp_id, M)
        stat_units = list(unit_map.keys())

        diff_change = _diff_around_criterion(
            states, targets_M, df_data, tau_change, criterion_mode
        )
        diff_no_change = _diff_during_zero_criterion(
            states, targets_M, df_data, tau_no_change
        )

        records[exp_id] = {
            ("hidden" if i == 0 else "cell"): {
                "step": np.abs(diff_change[:, unit]).mean(),
                "decay": np.abs(diff_no_change[:, unit]).mean(),
            }
            for i, unit in enumerate(stat_units)
        }

    df = pd.DataFrame.from_dict(
        {
            exp_id: pd.json_normalize(exp_data).iloc[0]
            for exp_id, exp_data in records.items()
        },
        orient="index",
    )
    df.columns = ["_".join(col.split(".")) for col in df.columns]
    return df

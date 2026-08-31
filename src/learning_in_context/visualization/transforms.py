"""Shared, memoized figure transformations for the paper's figure scripts.

Tier-2 of the three-tier figure pipeline: everything between the tier-1 cached
model-state artifacts (``data/cache/model_states/``, produced by ``dodo.py``'s
compute tasks) and a plot-ready DataFrame, which the tier-3 figure scripts turn
into styled SVG panels. Splitting the tiers is what lets styling iterate freely
without re-paying transformation cost. Figure scripts therefore render, they do
not compute: a panel needing data no ``dodo.py`` task produces gets a new
compute task rather than an inline computation (fig4's ElasticNet fits are the
one tolerated exception, and even those are memoized here).

Transforms are pure functions decorated with a single shared ``joblib.Memory``
instance so results are cached once and reused across every figure script.
``mo.persistent_cache`` is deliberately not used: it keys on per-cell identity,
which would defeat sharing a result between scripts.

Cache-dir seam
--------------
:func:`get_memory` is a factory. With ``cache_dir=None`` it reads the env var
``LIC_FIG_CACHE_DIR``; if that is unset it defaults to
``<repo_root>/data/cache/fig_transforms`` — the transform cache lives beside the
tier-1 artifacts under ``data/cache/``. The module-level :data:`MEMORY` is the
shared instance every memoized transform below is decorated with. Tests pass an
explicit ``cache_dir=`` (or set the env var) to isolate a throwaway cache.

joblib invalidates a cached result when the function's *source* or its arguments
change — editing a memoized function's body, docstring, or comments is enough to
force a recompute on the next run.

Keying discipline
-----------------
Every memoized transform keys on **paths / ids / params only** (dataset name,
model name, experiment id, small numeric knobs, hashable tuples) and loads the
underlying arrays internally — never accepting a preloaded ``states`` /
``samples`` / ``targets`` array as an argument. That keeps the cache key small
and stable instead of forcing joblib to hash multi-GB arrays on every call.

Promotion policy: a transform lives here once it has (or gains) a second
consumer; otherwise it stays script-local — still memoized, just not shared.
Promoting later is cheap.

Two fig5 transforms are exposed under stable names (the tests import both by
name):

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
_INTERVENTIONS_ROOT = _REPO_ROOT / "data" / "cache" / "interventions"


# --------------------------------------------------------------------------- #
# Memory infrastructure
# --------------------------------------------------------------------------- #
def get_memory(cache_dir: str | Path | None = None) -> joblib.Memory:
    """Build a ``joblib.Memory`` rooted at the fig-transform cache directory.

    Resolution order (evaluated at call time):
        1. explicit ``cache_dir`` argument,
        2. the ``LIC_FIG_CACHE_DIR`` environment variable,
        3. ``<repo_root>/data/cache/fig_transforms``.
    """
    if cache_dir is None:
        env = os.environ.get("LIC_FIG_CACHE_DIR")
        cache_dir = env if env else _DEFAULT_CACHE_DIR
    return joblib.Memory(location=str(cache_dir), verbose=0)


# Shared instance every memoized transform below is decorated with.
MEMORY = get_memory()

# Manual-recompute knob: FORCE_RECOMPUTE reflects the env var at
# import time. When set, the shared cache is cleared on import, so any run
# started with ``LIC_FIG_FORCE_RECOMPUTE=1`` recomputes every transform.
FORCE_RECOMPUTE = os.environ.get("LIC_FIG_FORCE_RECOMPUTE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
if FORCE_RECOMPUTE:
    MEMORY.clear(warn=False)


def clear_cache(memory: joblib.Memory | None = None) -> None:
    """Drop all memoized results — the manual-recompute knob.

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

# Contingency-block PROFILE unit set (see the fig5 script's docstring): the
# "hz" exemplar units restricted to the 6 contingency models. Confirmed
# numerically against figure 5 as composed for the paper — its contingency
# profile decays/steps match these units exactly (decay is
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
        # Contingent (bounce-triggered) color change. NOTE: the corresponding
        # cell in ``figures/fig_contingency_activity.py`` masks on
        # ``(targets[:, :, -4:-2] == 1).any(-1)`` (every bounce), but that cell
        # is dead code (empty "hz" dict) and its criterion does NOT reproduce
        # figure 5's contingency profile panel; ``targets[:, :, -2] == 1`` does,
        # exactly (verified numerically against the composed figure).
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
    samples, and targets internally (keyed only on ids/params),
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
def elasticnet_coefficient_paths(
    dataset: str,
    model_name: str,
    exp_id: str,
    stat: str,
    n_alphas: int = 50,
    logspace_hi: float = 0.0,
    logspace_lo: float = -6.0,
    M: int = 250,
    timestep_from_end: int = 50,
    l1_ratio_binary: float = 0.64,
    l1_ratio_linear: float = 0.4,
    max_iter_binary: int = 250,
    max_iter_linear: int = 1000,
) -> dict:
    """ElasticNet regularization-path fit for one decoder (fig4, memoized).

    Ports the fig4 data/fit pipeline (the linear regularization pipeline and its
    decoder/target definitions) from the exploratory analysis notebooks in the
    sibling ``hmdcpd-analysis`` repo. This is the pipeline's one tolerated piece
    of inline compute — the ElasticNet fits run here rather than in a ``dodo.py``
    task — so it is memoized through the shared :data:`MEMORY` and keyed only on
    paths/ids/params; the state array and target labels are loaded internally,
    never passed in.

    The decoder input ``X`` follows the source recipe: z-score the concatenation of the
    first ``M`` hidden and cell states over the (trial, time) axes, then take the
    single timestep ``timestep_from_end`` steps from the end — a
    ``(n_trials, 2 * hidden_size)`` feature matrix (32 columns: 16 hidden + 16
    cell). One coefficient per column gives the heatmap's 32 rows.

    Args:
        stat: ``"hz"`` (binary hazard-rate decode via elastic-net logistic
            regression → scalar F1) or ``"cont_r"`` (3-class contingency decode
            cast as elastic-net regression → per-label F1 vector, ``average=None``).
        n_alphas: number of points on the ``C``/alpha sweep.
        logspace_hi / logspace_lo: ``np.logspace`` exponents (``0`` → ``-6``,
            i.e. ``C`` from ``1`` down to ``1e-6``).

    Returns:
        A dict with:
          * ``coefs``: ``(n_units, n_alphas)`` array of per-alpha coefficients,
          * ``intercepts``: list of per-alpha intercept arrays,
          * ``metrics``: ``{"accuracy": (n_alphas,), "f1": (n_alphas,) or
            (n_alphas, n_labels)}`` — hz f1 is scalar-per-alpha, cont_r f1 is
            per-label (the composed figure's ``F1`` vs. ``F1 - Label 0/1/2``),
          * ``C_logspace``: the sweep values (x-axis, plotted as "ElasticNet Alpha").
    """
    from sklearn.linear_model import ElasticNet, LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    # --- decoder input X ----------------------------------------------------- #
    path = _dataset_dir(dataset) / model_name / f"{exp_id}.npz"
    model_data = np.load(str(path), allow_pickle=True)
    z = stats.zscore(
        np.concatenate(
            [model_data["hiddens"][:, :M], model_data["cells"][:, :M]], axis=-1
        ),
        axis=(0, 1),
        ddof=1,
    )
    X = z[:, -timestep_from_end]  # (n_trials, 2 * hidden_size)

    # --- decoder target y --------------------------------------------------- #
    df_data = pd.read_csv(_dataset_dir(dataset) / "trial_meta.csv", index_col=0)
    if stat == "hz":
        y = df_data["Hazard Rate"].eq("High").astype(int).values
    elif stat == "cont_r":
        y = df_data["Contingency"].map({"Low": 0, "Medium": 1, "High": 2}).values
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unknown stat {stat!r}")

    C_logspace = np.logspace(logspace_hi, logspace_lo, n_alphas)

    coefs: list = []
    intercepts: list = []
    acc_path: list = []
    f1_path: list = []

    if stat == "cont_r":
        # Regression cast: map the 0/1/2 labels to [0, .5, 1], fit ElasticNet,
        # predict the nearest label. alpha comes from the source's C→alpha map.
        y_choices = np.unique(y)
        y_reg = (y_choices - y_choices.min()) / (y_choices.max() - y_choices.min())
        y_scaled = (y - y_choices.min()) / (y_choices.max() - y_choices.min())
        for C in C_logspace:
            # The C->alpha map hits exactly 0 at C == 1 (the no-regularization
            # endpoint). Under the sklearn the source notebooks ran against that
            # behaved as OLS; sklearn 1.8's ElasticNet(alpha=0) collapses to a constant fit
            # (the "coordinate descent with no regularization" path), corrupting
            # the leftmost alpha of the contingency panels. Floor alpha to a
            # tiny positive value to reproduce the intended near-OLS behavior.
            alpha_val = max(-(1 - 1 / C) / 100000, 1e-9)
            reg = ElasticNet(
                l1_ratio=l1_ratio_linear,
                max_iter=max_iter_linear,
                alpha=alpha_val,
            ).fit(X, y_scaled)
            pred = np.argmin(
                np.abs(reg.predict(X).reshape((-1, 1)) - y_reg), axis=-1
            )
            coefs.append(reg.coef_.ravel())
            intercepts.append(np.reshape(reg.intercept_, (1,)))
            acc_path.append(accuracy_score(y, pred))
            f1_path.append(f1_score(y, pred, average=None))
    else:  # "hz" — binary elastic-net logistic decode.
        for C in C_logspace:
            reg = LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                l1_ratio=l1_ratio_binary,
                max_iter=max_iter_binary,
                C=C,
            ).fit(X, y)
            pred = reg.predict(X)
            coefs.append(reg.coef_.ravel())
            intercepts.append(reg.intercept_)
            acc_path.append(accuracy_score(y, pred))
            f1_path.append(f1_score(y, pred))  # binary → scalar

    return {
        "coefs": np.stack(coefs, axis=-1),  # (n_units, n_alphas)
        "intercepts": intercepts,
        "metrics": {
            "accuracy": np.array(acc_path),
            "f1": np.array(f1_path),
        },
        "C_logspace": C_logspace,
    }


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
            contingency models; both contingency profile blocks, verified
            against the composed figure), or ``"cont"`` (contingency exemplar units, used
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


# --------------------------------------------------------------------------- #
# Intervention prediction frames (shared by fig6 and fig7)
# --------------------------------------------------------------------------- #
# The per-model, per-alpha intervention prediction frames feed fig6's
# intervention time-courses and summary point plots, and are reused by fig7's
# gate panels — hence a single memoized transform here rather than a copy per
# figure script. Ported verbatim (sample windowing + per-model prediction
# frame) from the exploratory analysis notebooks in the sibling
# ``hmdcpd-analysis`` repo, split into this memoized per-model transform so
# figure scripts only pay the load/window cost once. Keyed on ids/params only;
# the ``.npz`` preds array and trial metadata are loaded internally, never
# passed in.


def _interventions_meta() -> pd.DataFrame:
    """Trial metadata shared across every intervention model."""
    return pd.read_csv(_INTERVENTIONS_ROOT / "trial_meta.csv", index_col=0)


def _intervention_npz_name(stat: str, unit: str | None, num_alphas: int) -> str:
    """Reconstruct the cached intervention-prediction npz filename.

    ``{stat}[-{unit}]-centroid-interventions-{num_alphas}-alphas.npz`` where
    ``stat`` is ``"hz"`` / ``"cont"`` and ``unit`` is ``"hidden"`` / ``"cell"``
    (or ``None`` for the both-units frame).
    """
    parts = [stat]
    if unit is not None:
        parts.append(unit)
    parts += ["centroid-interventions", str(num_alphas), "alphas.npz"]
    return "-".join(parts)


def _window_samples(samples: np.ndarray, endpoints: np.ndarray, N: int) -> np.ndarray:
    """Take the last ``N`` timesteps before each trial's endpoint.

    Ported verbatim from the source analysis notebooks.
    """
    b, T, f = samples.shape
    t_idx = endpoints[:, None] + np.arange(-N, 0)
    b_idx = np.arange(b)[:, None]
    return samples[b_idx, t_idx, :]


@MEMORY.cache
def intervention_prediction_frame(
    model_name: str,
    exp_id: str,
    stat: str,
    unit: str | None,
    num_alphas: int = 11,
    N: int = 26,
) -> pd.DataFrame:
    """Tidy per-model intervention prediction frame (fig6/fig7, memoized).

    Loads one model's centroid-intervention predictions
    (``interventions/{model_name}/{exp_id}/{stat}[-{unit}]-...alphas.npz``),
    windows the last ``N`` timesteps before each trial endpoint, extracts the
    probability of the entered colour per (alpha, video, timestep, centroid),
    and returns the melted long frame the intervention panels consume.

    Args:
        model_name: sub-directory under ``interventions/`` (e.g. ``"lstm"``).
        exp_id: model id (e.g. ``"san-4604"``).
        stat: ``"hz"`` (hazard-rate) or ``"cont"`` (contingency).
        unit: ``"hidden"`` / ``"cell"`` (single-unit intervention) or ``None``
            (both-units intervention).
        num_alphas: number of intervention strengths (11 → 0.0-1.0).
        N: timesteps to window (26 for hazard, 24 for contingency).

    Returns:
        Long DataFrame with columns
        ``[Alpha, Video, Hazard Rate, idx_time, Contingency, trial, Timestep,
        Value, Centroid]``. ``Value`` is P(entered colour changes) = ``1 - p``.
    """
    df_data = _interventions_meta()
    alphas = np.linspace(0, 1, num_alphas)
    name = _intervention_npz_name(stat, unit, num_alphas)
    path = _INTERVENTIONS_ROOT / model_name / exp_id / name
    preds = np.load(str(path))["preds"]

    color_entered = df_data["color_entered"].values - 1
    lengths = df_data["length"].values

    preds_list = []
    for centroid_idx in range(2):
        centroid_preds = preds[:, centroid_idx]
        windowed_preds = [
            _window_samples(centroid_preds[alpha_idx], lengths, N)
            for alpha_idx in range(num_alphas)
        ]
        preds_list.append(np.stack(windowed_preds))
    preds_int = np.stack(preds_list)
    _, _, num_videos, timesteps, _num_channels = preds_int.shape

    frames = []
    for i, preds_norm in enumerate(preds_int):
        pred_same_color = preds_norm[
            np.arange(num_alphas)[:, None, None],
            np.arange(num_videos)[None, :, None],
            np.arange(timesteps)[None, None, :],
            color_entered[None, :, None],
        ]
        pred_same_color_reshaped = pred_same_color.reshape(-1, timesteps)
        df_preds = pd.DataFrame(pred_same_color_reshaped)
        df_preds["Alpha"] = np.repeat(alphas, num_videos)
        df_preds["Video"] = list(range(num_videos)) * num_alphas
        df_preds["Hazard Rate"] = list(df_data["Hazard Rate"].values) * num_alphas
        df_preds["idx_time"] = list(df_data["idx_time"].values) * num_alphas
        df_preds["Contingency"] = list(df_data["Contingency"].values) * num_alphas
        df_preds["trial"] = list(df_data["trial"].values) * num_alphas
        melted = df_preds.melt(
            id_vars=["Alpha", "Video", "Hazard Rate", "idx_time", "Contingency", "trial"],
            var_name="Timestep",
            value_name="Value",
        )
        melted["Value"] = 1 - melted["Value"]
        melted["Centroid"] = i
        frames.append(melted)

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# fig7 — gate-rescue intervention frames + delta-gate-activity scatters
# --------------------------------------------------------------------------- #
# Ported (internals verbatim) from the exploratory analysis notebooks in the
# sibling ``hmdcpd-analysis`` repo: the gate-frozen "rescue" interventions and
# the per-model / per-unit delta gate activity. Both read the "all-states"
# intervention caches under ``data/cache/interventions/`` and are keyed on
# ids/params only; arrays are loaded internally, never passed in.


def _gate_frozen_npz_name(
    stat: str, unit: str | None, gate: tuple[str, ...], num_alphas: int
) -> str:
    """Reconstruct the cached gate-frozen intervention npz name.

    ``{stat}[-{unit}]-{load_gate}-gate[s]-frozen-centroid-interventions-
    {num_alphas}-all-states-alphas.npz`` where ``load_gate`` is the gate letters
    sorted and joined (e.g. ``('i','f')`` → ``"fi"``) and the ``-s`` suffix on
    ``gate`` appears only for multi-gate freezes.
    """
    load_gate = "".join(sorted(gate))
    frozen = f"gate{'s' if len(load_gate) > 1 else ''}-frozen"
    parts = [stat]
    if unit is not None:
        parts.append(unit)
    parts += [
        load_gate,
        frozen,
        "centroid-interventions",
        str(num_alphas),
        "all-states",
        "alphas.npz",
    ]
    return "-".join(parts)


@MEMORY.cache
def gate_rescue_prediction_frame(
    model_name: str,
    exp_id: str,
    stat: str,
    unit: str | None,
    gate: tuple[str, ...],
    num_alphas: int = 11,
    N: int = 29,
) -> pd.DataFrame:
    """Tidy per-model gate-frozen ("rescue") intervention frame (fig7, memoized).

    Same shape as :func:`intervention_prediction_frame`, but reads the
    gate-frozen "all-states" cache
    (``interventions/{model_name}/{exp_id}/{stat}-{unit}-{load_gate}-gates-
    frozen-...-all-states-alphas.npz``) where the named gates are held at their
    control value while the centroid intervention runs. Ported verbatim from the
    source notebooks' ``preds`` → melted-frame path; ``states``/``gates`` in the
    npz are ignored here (they feed the separate gate-activity transform).

    Args:
        gate: gate letters to freeze (e.g. ``("i", "f")``). Passed as a tuple so
            the cache key stays hashable.
        num_alphas: number of intervention strengths (11).
        N: timesteps to window before each trial endpoint (29 for the gate-rescue cache).

    Returns:
        Long DataFrame with the same columns as
        :func:`intervention_prediction_frame`.
    """
    df_data = _interventions_meta()
    alphas = np.linspace(0, 1, num_alphas)
    name = _gate_frozen_npz_name(stat, unit, tuple(gate), num_alphas)
    path = _INTERVENTIONS_ROOT / model_name / exp_id / name
    preds = np.load(str(path))["preds"]

    color_entered = df_data["color_entered"].values - 1
    lengths = df_data["length"].values

    preds_list = []
    for centroid_idx in range(2):
        centroid_preds = preds[:, centroid_idx]
        windowed_preds = [
            _window_samples(centroid_preds[alpha_idx], lengths, N)
            for alpha_idx in range(num_alphas)
        ]
        preds_list.append(np.stack(windowed_preds))
    preds_int = np.stack(preds_list)
    _, _, num_videos, timesteps, _num_channels = preds_int.shape

    frames = []
    for i, preds_norm in enumerate(preds_int):
        pred_same_color = preds_norm[
            np.arange(num_alphas)[:, None, None],
            np.arange(num_videos)[None, :, None],
            np.arange(timesteps)[None, None, :],
            color_entered[None, :, None],
        ]
        pred_same_color_reshaped = pred_same_color.reshape(-1, timesteps)
        df_preds = pd.DataFrame(pred_same_color_reshaped)
        df_preds["Alpha"] = np.repeat(alphas, num_videos)
        df_preds["Video"] = list(range(num_videos)) * num_alphas
        df_preds["Hazard Rate"] = list(df_data["Hazard Rate"].values) * num_alphas
        df_preds["idx_time"] = list(df_data["idx_time"].values) * num_alphas
        df_preds["Contingency"] = list(df_data["Contingency"].values) * num_alphas
        df_preds["trial"] = list(df_data["trial"].values) * num_alphas
        melted = df_preds.melt(
            id_vars=["Alpha", "Video", "Hazard Rate", "idx_time", "Contingency", "trial"],
            var_name="Timestep",
            value_name="Value",
        )
        melted["Value"] = 1 - melted["Value"]
        melted["Centroid"] = i
        frames.append(melted)

    return pd.concat(frames, ignore_index=True)


@MEMORY.cache
def gate_activity_delta_frame(
    model_name: str,
    exp_ids: tuple[str, ...],
    num_alphas: int = 11,
    len_gray: int = 24,
    gate_order: tuple[str, ...] = ("i", "f", "g", "o"),
) -> pd.DataFrame:
    """Per-model, per (color_entered × unit) delta gate activity (fig7, memoized).

    For every model, the signed change in each LSTM gate's per-unit activity between the two extreme
    centroid interventions (alpha 0 → 1), oriented so the delta always reads
    "High Hz minus Low Hz" — i.e. toward the target hazard rate. Both fig7 gate
    scatters derive from this single frame:

    * the single-model scatter is ``frame[frame.model == exemplar]`` (48 rows =
      3 colours × 16 units),
    * the aggregated unit-mean scatter is
      ``frame.groupby(["color_entered", "model"])[["i","f","g","o"]].mean()``
      (30 rows = 3 colours × 10 models).

    Ported verbatim from the source notebook's per-model intervention build,
    delta computation, and per-unit melt/merge. Only the plain "all-states"
    cell-unit cache
    (``hz-cell-centroid-interventions-{num_alphas}-all-states-alphas.npz``) is
    read; only its ``gates`` array is loaded (the ~4 GB per model), keyed on
    ids/params.

    The restriction to ``idx_time == 2`` trials, ``alpha ∈ {0, 1}`` and the last
    ``len_gray`` valid timesteps is applied *before* materializing rows — this is
    output-identical to the source (which builds the full frame then filters)
    but avoids holding hundreds of thousands of unused rows.

    Returns:
        DataFrame with columns ``[color_entered, unit_idx, i, f, g, o, model]``.
    """
    meta = _interventions_meta()
    lengths = meta["length"].values
    hazard = meta["Hazard Rate"].values
    color_entered_idx = meta["color_entered"].values - 1
    array_color = np.array(["Red", "Green", "Blue"])
    idx_time2 = meta["idx_time"].values == 2

    # centroid index → intervention target hazard rate.
    dict_ints_name = {0: "Low", 1: "High"}
    alpha_idxs = [0, num_alphas - 1]

    per_model = []
    for exp_id in exp_ids:
        name = f"hz-cell-centroid-interventions-{num_alphas}-all-states-alphas.npz"
        gates = np.load(str(_INTERVENTIONS_ROOT / model_name / exp_id / name))["gates"]
        _num_alphas, _num_cent, _batch, timesteps, gate_width = gates.shape
        num_units = gate_width // len(gate_order)

        columns_gates = [
            f"{g.upper()}{u}" for g in gate_order for u in range(num_units)
        ]
        dict_column_gates = {
            g: [c for c in columns_gates if c.startswith(g.upper())]
            for g in gate_order
        }

        # Valid, grayzone, idx_time==2 (batch, time) coordinates (shared across
        # every alpha/centroid slice, so both alpha subframes align positionally).
        tt = np.arange(timesteps)
        remaining = lengths[:, None] - tt[None, :]
        valid = (
            (tt[None, :] < lengths[:, None])
            & idx_time2[:, None]
            & (remaining < len_gray)
        )
        batch_idx, time_idx = np.where(valid)

        blocks = []
        for centroid_idx, target_hz in dict_ints_name.items():
            for alpha_idx in alpha_idxs:
                g_slice = gates[alpha_idx, centroid_idx]  # (batch, time, gate_width)
                data = {"batch": batch_idx, "timestep": time_idx}
                for gi, gate in enumerate(gate_order):
                    block = g_slice[batch_idx, time_idx, gi * num_units : (gi + 1) * num_units]
                    for u in range(num_units):
                        data[f"{gate.upper()}{u}"] = block[:, u]
                df = pd.DataFrame(data)
                df["color_entered"] = array_color[color_entered_idx[batch_idx]]
                df["Hazard Rate"] = hazard[batch_idx]
                df["timesteps_remaining"] = lengths[batch_idx] - time_idx
                df["alpha"] = alpha_idx / 10
                df["target_hz"] = target_hz
                blocks.append(df)
        df_ints = pd.concat(blocks, ignore_index=True)
        del gates

        # Signed delta toward the target hazard rate.
        list_dfs = []
        for (hz, target_hz), df_hz in df_ints.groupby(["Hazard Rate", "target_hz"]):
            if hz == target_hz:
                continue
            subs = [d for _, d in df_hz.groupby("alpha")]  # ascending: alpha0, alpha1
            df_a0, df_a1 = subs[0], subs[1]
            assert (
                df_a0[["batch", "timestep"]].to_numpy()
                == df_a1[["batch", "timestep"]].to_numpy()
            ).all(), "alpha subframes misaligned"
            df_delta = df_a0.copy().drop("alpha", axis=1)
            if hz == "Low" and target_hz == "High":
                df_delta[columns_gates] = (
                    df_a1[columns_gates].to_numpy() - df_a0[columns_gates].to_numpy()
                )
            elif hz == "High" and target_hz == "Low":
                df_delta[columns_gates] = (
                    df_a0[columns_gates].to_numpy() - df_a1[columns_gates].to_numpy()
                )
            else:  # pragma: no cover - guarded above
                raise ValueError("Invalid combination")
            list_dfs.append(df_delta)
        df_delta_ints = pd.concat(list_dfs)

        # Per-colour mean over units → one row per (colour, unit).
        mean_df = df_delta_ints.groupby("color_entered")[columns_gates].mean()
        gate_dfs = []
        for gate, cols in dict_column_gates.items():
            gate_data = (
                mean_df[cols]
                .melt(var_name="unit", value_name=gate, ignore_index=False)
                .reset_index()
            )
            gate_data["unit_idx"] = gate_data["unit"].str.extract(r"(\d+)").astype(int)
            gate_data = gate_data.drop("unit", axis=1)
            gate_dfs.append(gate_data)
        plot_df = gate_dfs[0]
        for gate_data in gate_dfs[1:]:
            plot_df = pd.merge(plot_df, gate_data, on=["color_entered", "unit_idx"])
        plot_df["model"] = exp_id
        per_model.append(plot_df)

    return pd.concat(per_model).reset_index(drop=True)

"""Contract tests for ``figures/fig3_task_results.py`` (marimo, dual-use).

Figure 3 is a 2x3 grid rendered as SIX individual panels under
``figures/panels/fig3/``:

* Top row ("Straight Path Trials") — CWC vs. Grayzone Position, split by
  Hazard Rate (Low/High), one panel per data source:
  ``cwc_straight_participants.svg``, ``cwc_straight_rnn.svg``,
  ``cwc_straight_lstm.svg``. Every panel is self-contained, so all three carry
  a Hazard Rate legend — including the participants panel — rather than only
  the rightmost one.
* Bottom row ("Wall Bounce Trials") — CWC vs. Contingency, a single red
  family with no hue split and no legend (the x tick labels already name the
  levels): ``cwc_bounce_participants.svg``, ``cwc_bounce_rnn.svg``,
  ``cwc_bounce_lstm.svg``.

Data sources:

* RNN/LSTM columns go through ``transforms.model_cwc_by_hazard`` /
  ``transforms.model_cwc_by_contingency`` (see
  ``tests/test_model_cwc_transforms.py`` for those transforms' own contract),
  restricted per panel to a single model type.
* The participants column goes through
  ``learning_in_context.analysis.participants.participant_cwc_by_hazard`` /
  ``participant_cwc_by_contingency``, fed a ``{participant_id: responses}``
  dict rebuilt from the ``participant_cwc.parquet`` artifact and the FULL
  straight/bounce trial metadata (never a per-participant subset — grouping
  every participant against the same buckets is what makes the panel
  comparable across participants). The script renders only; it reads the
  ``data/cache/participants/`` artifacts produced by the participant_stats
  pipeline rather than rerunning it.
* Both sides render through the shared ``cwc_plots.plot_cwc_swarm`` renderer.

This file pins four tiers:

1. Static script structure (AST-level, no data, no subprocess) — the script
   exists, is a marimo app, imports the right modules, avoids computing
   inside the script what belongs in ``transforms``/``participants``, names
   all six panels, and does not import the sibling ``hmdcpd-analysis`` repo.
2. Content contracts for the two data sources' condition grouping, checked
   against the transforms' and participants' actual output (real cached
   artifacts already exist on disk in this repo) rather than against
   rendered SVGs.
3. Style contracts — legend presence on the hazard row (including the
   participants panel) vs. its absence on the contingency row, and dashed
   gridlines somewhere in the render path.
4. A headless-run tier (``python figures/fig3_task_results.py``) that lands
   exactly the six expected SVGs, each carrying live text, and reproduces
   byte-identical output across two runs that each get their own fresh
   transform cache directory (real determinism, not a warm-cache artifact).

Resolved ambiguities:

1. ``model_cwc_by_hazard``/``model_cwc_by_contingency`` may be called once
   per model type (``model_types=("rnn",)`` / ``("lstm",)``) or once with
   both and filtered afterwards (``model_types=("rnn", "lstm")``) — this file
   does not require one shape over the other. What it pins is the union of
   ``model_types`` resolved across every call to each transform: it must
   cover exactly ``{"rnn", "lstm"}`` (never ``"ibo"``, which is fig2's
   territory).
2. ``num_participants`` must come from the participant-cohort count artifact
   (``data/cache/participants/participant_counts.json``'s ``final_n``), not
   a literal integer — every call's ``num_participants`` argument is checked
   to NOT resolve to a literal int (directly or via a module constant bound
   to a literal), and the script's source must reference the counts artifact
   by name.
3. The model-side dataset name is pinned explicitly to ``"participant_dataset"``
   (unlike fig2's ``CWC_DATASET``, there is no second candidate here — fig3's
   model panels sample a pool matched to the human cohort, which only that
   dataset provides).
4. Panels are pinned as six statically-resolvable calls (one ``save_panel``
   call and one ``plot_cwc_swarm`` call per panel, each with a literal or
   module-constant panel name / model_types / hue), not a shared loop over
   the RNN/LSTM columns the way fig5 loops over its unit blocks — fig5's
   loop reuses one identical rendering recipe per iteration, while fig3's
   three columns (participants, RNN, LSTM) differ enough in data-sourcing
   (the participants column goes through a different module entirely) that
   an unrolled, one-cell-per-panel shape reads more clearly and is what this
   file's static checks (panel-name / model_types / hue resolution) require.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import pandas as pd

    from learning_in_context.analysis import participants
    from learning_in_context.visualization import transforms

    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - only if a landed module breaks
    pd = None  # type: ignore[assignment]
    participants = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG3_SCRIPT = REPO_ROOT / "figures" / "fig3_task_results.py"
PANELS_DIR = REPO_ROOT / "figures" / "panels" / "fig3"
MODEL_STATES_DIR = REPO_ROOT / "data" / "cache" / "model_states"
PARTICIPANTS_DIR = REPO_ROOT / "data" / "cache" / "participants"

PANEL_BASENAMES = [
    "cwc_straight_participants",
    "cwc_straight_rnn",
    "cwc_straight_lstm",
    "cwc_bounce_participants",
    "cwc_bounce_rnn",
    "cwc_bounce_lstm",
]
EXPECTED_PANELS = [f"{name}.svg" for name in PANEL_BASENAMES]

_PARTICIPANT_DATASET = "participant_dataset"
_MODEL_TYPES = {"rnn", "lstm"}

# --------------------------------------------------------------------------- #
# Real-artifact readiness (module-level, so tests skip cleanly rather than
# erroring when this repo's cache is not populated).
# --------------------------------------------------------------------------- #


def _dataset_ready(name: str) -> bool:
    d = MODEL_STATES_DIR / name
    if not ((d / "trial_meta.csv").exists() and (d / "samples.npy").exists()):
        return False
    return all((d / model_type).is_dir() for model_type in _MODEL_TYPES)


_HAS_PARTICIPANT_MODEL_DATA = _dataset_ready(_PARTICIPANT_DATASET)

_ARTIFACT_NAMES = ("participant_stats_filtered.parquet", "participant_cwc.parquet", "participant_counts.json")
_HAS_PARTICIPANTS_ARTIFACTS = all((PARTICIPANTS_DIR / name).exists() for name in _ARTIFACT_NAMES)

_COUNTS_PATH = PARTICIPANTS_DIR / "participant_counts.json"
if _COUNTS_PATH.exists():
    _NUM_PARTICIPANTS: int | None = json.loads(_COUNTS_PATH.read_text())["final_n"]
else:  # pragma: no cover - only when the artifact is absent
    _NUM_PARTICIPANTS = None


# --------------------------------------------------------------------------- #
# AST plumbing (self-contained — does not import from tests/test_fig2_panels.py,
# which is owned by a sibling worker and may change independently).
# --------------------------------------------------------------------------- #
def _parse() -> ast.Module:
    return ast.parse(FIG3_SCRIPT.read_text(), filename=str(FIG3_SCRIPT))


def _source() -> str:
    return FIG3_SCRIPT.read_text()


def _module_constants(tree: ast.Module) -> dict[str, object]:
    """Top-level ``NAME = <literal>`` assignments, resolved via ``literal_eval``.

    Scans every ``Assign`` in the module (not just the top level) because
    marimo cells are functions — a script-level constant lives inside a cell
    function's body, not at module scope.
    """
    constants: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError, TypeError):
                    pass
    return constants


def _resolve(value_node: ast.expr | None, constants: dict[str, object]):
    """Resolve a call-argument AST node to a literal Python value, or ``None``.

    Direct literals resolve via ``ast.literal_eval``. A bare ``Name`` resolves
    through a module-level ``NAME = <literal>`` constant assignment found
    anywhere in the script.
    """
    if value_node is None:
        return None
    try:
        return ast.literal_eval(value_node)
    except (ValueError, SyntaxError, TypeError):
        pass
    if isinstance(value_node, ast.Name) and value_node.id in constants:
        return constants[value_node.id]
    return None


def _calls_to(tree: ast.Module, attr_name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr_name
    ]


def _kwarg_nodes(call: ast.Call) -> dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}


def _arg_value(call: ast.Call, pos: int, name: str) -> ast.expr | None:
    """The AST node for a positional-or-keyword argument, wherever it lands."""
    if len(call.args) > pos:
        return call.args[pos]
    return _kwarg_nodes(call).get(name)


def _build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _has_loop_ancestor(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """True if ``node`` sits inside a ``for``/``while`` or a comprehension.

    Used to catch the "called once per participant in a loop" anti-pattern —
    the participant CWC calls must run once against the full metadata, not
    once per participant against a subset.
    """
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.For, ast.AsyncFor, ast.While)):
            return True
        if isinstance(current, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return True
        current = parents.get(current)
    return False


def _imports_module_as(tree: ast.Module, module_substring: str, bound_name: str) -> bool:
    """True if the script imports a module (whose dotted path contains
    ``module_substring``) and binds it to ``bound_name`` — the
    ``from learning_in_context.xxx import yyy`` / ``import ... as yyy`` house
    style used consistently by every other figure script for ``transforms``,
    ``cwc_plots``, and ``paper_style``.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is None or module_substring not in node.module:
                continue
            for alias in node.names:
                if (alias.asname or alias.name) == bound_name:
                    return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if module_substring not in alias.name:
                    continue
                bound = (alias.asname or alias.name).rsplit(".", 1)[-1]
                if bound == bound_name:
                    return True
    return False


def _is_int_literal(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# --------------------------------------------------------------------------- #
# Tier 0 — script exists
# --------------------------------------------------------------------------- #
class TestScriptExists:
    def test_script_exists(self):
        assert FIG3_SCRIPT.exists(), f"figure script not found: {FIG3_SCRIPT}"


# --------------------------------------------------------------------------- #
# Tier 1 — static script structure
# --------------------------------------------------------------------------- #
class TestScriptStructure:
    def test_defines_a_marimo_app(self):
        tree = _parse()
        source = _source()
        assert "import marimo" in source
        assigns_app = any(
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "app" for t in node.targets)
            for node in ast.walk(tree)
        )
        assert assigns_app, "expected a module-level `app = marimo.App(...)` assignment"

    def test_imports_transforms_module(self):
        assert _imports_module_as(_parse(), "visualization", "transforms")

    def test_imports_cwc_plots_module(self):
        assert _imports_module_as(_parse(), "visualization", "cwc_plots")

    def test_imports_paper_style_module(self):
        assert _imports_module_as(_parse(), "visualization", "paper_style")

    def test_imports_participants_module(self):
        assert _imports_module_as(_parse(), "analysis", "participants"), (
            "expected a `from learning_in_context.analysis import participants` "
            "import — the house style used for every other shared module in "
            "this pipeline (transforms, cwc_plots, paper_style are all bound "
            "as modules, not individual functions)"
        )

    def test_defines_save_svgs_switch(self):
        tree = _parse()
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "save_svgs" for t in node.targets):
                continue
            call = node.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "switch"
            ):
                found = True
                break
        assert found, (
            "expected a `save_svgs = mo.ui.switch(...)` assignment; the full "
            "four-part gating contract is enforced by tests/test_fig_save_toggle.py"
        )

    def test_does_not_directly_load_cache_arrays(self):
        """Tier-2 loads (``np.load`` / ``pd.read_csv``) belong in transforms.

        The participants artifacts are the one sanctioned exception to
        "render, don't load" — but they are parquet/JSON
        (``pd.read_parquet`` / ``json.load``), not ``np.load``/``pd.read_csv``,
        so forbidding exactly those two calls already carves out that
        exception without any special-casing.
        """
        tree = _parse()
        offenders = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            root = node.func.value
            root_name = root.id if isinstance(root, ast.Name) else None
            if root_name in {"np", "numpy"} and node.func.attr == "load":
                offenders.append(node.lineno)
            if root_name in {"pd", "pandas"} and node.func.attr == "read_csv":
                offenders.append(node.lineno)
        assert not offenders, (
            f"figures/fig3_task_results.py loads cache arrays directly at "
            f"line(s) {offenders}; route this through "
            "learning_in_context.visualization.transforms instead"
        )

    def test_save_panel_calls_name_exactly_the_six_expected_panels(self):
        tree = _parse()
        constants = _module_constants(tree)
        calls = _calls_to(tree, "save_panel")
        assert calls, "expected at least one call to paper_style.save_panel(...)"

        names: set[str] = set()
        unresolved = []
        for call in calls:
            name_node = call.args[2] if len(call.args) >= 3 else _kwarg_nodes(call).get("name")
            value = _resolve(name_node, constants)
            if isinstance(value, str):
                names.add(value)
            else:
                unresolved.append(call.lineno)

        assert not unresolved, (
            f"save_panel call(s) at line(s) {unresolved} have a `name` "
            "argument that could not be resolved to a literal or a "
            "module-level constant"
        )
        assert names == set(PANEL_BASENAMES), (
            f"save_panel call(s) name {sorted(names)}, expected exactly "
            f"{sorted(PANEL_BASENAMES)}"
        )

    def test_plot_cwc_swarm_is_called_exactly_six_times(self):
        calls = _calls_to(_parse(), "plot_cwc_swarm")
        assert len(calls) == 6, (
            f"expected exactly 6 calls to cwc_plots.plot_cwc_swarm (one per "
            f"panel); found {len(calls)} — the CWC panels must go through "
            "the landed renderer, one call per self-contained panel"
        )

    def test_does_not_import_the_sibling_analysis_repo(self):
        assert "hmdcpd" not in _source()


# --------------------------------------------------------------------------- #
# Tier 1a-2 — render-only rule: read the landed artifacts, never rerun the
# participant pipeline that produces them.
# --------------------------------------------------------------------------- #
_FORBIDDEN_PIPELINE_CALLEES = {
    "run_participant_pipeline",
    "write_participant_stats",
    "load_all_participant_responses",
    "load_participant_responses",
    "load_all_participant_demographics",
    "compute_initial_stats_all_participants",
    "compute_participant_cwc",
}


class TestRenderOnlyRule:
    def test_does_not_rerun_the_participant_pipeline(self):
        """The script reads ``data/cache/participants/`` artifacts; it never
        recomputes them by re-loading raw jsPsych exports or rescoring CWC.

        ``load_trial_metadata`` is deliberately not forbidden here — it reads
        ``trial_meta.csv``/``dataset_meta.pkl``, not raw participant exports,
        and the script may legitimately need the full trial metadata.
        """
        tree = _parse()
        offenders = [
            (node.lineno, node.func.attr)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _FORBIDDEN_PIPELINE_CALLEES
        ]
        assert not offenders, (
            f"found call(s) to participant-pipeline function(s) {offenders} — "
            "the figure script must read the landed data/cache/participants/ "
            "artifacts, never rerun the pipeline that produces them"
        )

    def test_asserts_on_missing_participant_artifacts_with_a_clear_message(self):
        """Per the render-only rule: assert on missing artifacts, don't
        silently proceed. Non-brittle: looks for an ``assert`` whose test
        calls ``.exists()`` and whose message is a non-empty string/f-string,
        without pinning the exact wording.
        """
        tree = _parse()
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert) or node.msg is None:
                continue
            has_exists_check = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "exists"
                for n in ast.walk(node.test)
            )
            if not has_exists_check:
                continue
            if isinstance(node.msg, ast.Constant) and isinstance(node.msg.value, str) and node.msg.value.strip():
                found = True
                break
            if isinstance(node.msg, ast.JoinedStr):  # an f-string message
                found = True
                break
        assert found, (
            "expected an `assert <path>.exists(), \"<clear message>\"` "
            "guarding the participant artifacts — the render-only rule "
            "requires failing clearly on missing artifacts, not silently "
            "proceeding or falling back to rerunning the pipeline"
        )


# --------------------------------------------------------------------------- #
# Tier 1b — model-side (RNN/LSTM) CWC call contracts
# --------------------------------------------------------------------------- #
class TestModelCwcCallContracts:
    def _cwc_calls(self) -> tuple[list[ast.Call], list[ast.Call]]:
        tree = _parse()
        return (
            _calls_to(tree, "model_cwc_by_hazard"),
            _calls_to(tree, "model_cwc_by_contingency"),
        )

    def test_at_least_one_call_each_to_the_model_cwc_transforms(self):
        hazard_calls, contingency_calls = self._cwc_calls()
        assert hazard_calls, "expected at least one call to model_cwc_by_hazard"
        assert contingency_calls, "expected at least one call to model_cwc_by_contingency"

    def test_model_dataset_constant_is_participant_dataset(self):
        constants = _module_constants(_parse())
        matching = [name for name, value in constants.items() if value == _PARTICIPANT_DATASET]
        assert matching, (
            "expected a module-level string constant equal to "
            f"'{_PARTICIPANT_DATASET}' naming the model-side dataset (fig3's "
            "model panels sample a pool matched to the human cohort, so this "
            "dataset is not an open call the way fig2's CWC_DATASET is)"
        )

    def test_model_cwc_calls_reference_the_dataset_constant_not_a_literal(self):
        tree = _parse()
        constants = _module_constants(tree)
        dataset_constant_names = {
            name for name, value in constants.items() if value == _PARTICIPANT_DATASET
        }
        hazard_calls, contingency_calls = self._cwc_calls()
        for call in (*hazard_calls, *contingency_calls):
            dataset_node = _arg_value(call, 0, "dataset")
            assert dataset_node is not None, (
                f"line {call.lineno}: call to {call.func.attr} does not pass a "
                "`dataset` argument"
            )
            assert (
                isinstance(dataset_node, ast.Name) and dataset_node.id in dataset_constant_names
            ), (
                f"line {call.lineno}: `dataset` is not a reference to the "
                f"'{_PARTICIPANT_DATASET}' constant — the dataset name must "
                "not be hardcoded independently in each call"
            )

    def test_model_cwc_calls_use_only_rnn_and_lstm_model_types(self):
        tree = _parse()
        constants = _module_constants(tree)
        hazard_calls, contingency_calls = self._cwc_calls()
        for call in (*hazard_calls, *contingency_calls):
            model_types = _resolve(_arg_value(call, 1, "model_types"), constants)
            assert model_types is not None and set(model_types) <= _MODEL_TYPES, (
                f"line {call.lineno}: expected model_types drawn only from "
                f"{sorted(_MODEL_TYPES)} (fig3 is RNN/LSTM only — 'ibo' is "
                f"fig2's territory); resolved {model_types!r}"
            )
            assert model_types, f"line {call.lineno}: model_types resolved empty"

    def test_hazard_calls_cover_both_rnn_and_lstm(self):
        tree = _parse()
        constants = _module_constants(tree)
        hazard_calls, _ = self._cwc_calls()
        covered: set[str] = set()
        for call in hazard_calls:
            model_types = _resolve(_arg_value(call, 1, "model_types"), constants)
            if model_types:
                covered |= set(model_types)
        assert covered == _MODEL_TYPES, (
            f"the model_cwc_by_hazard call(s) cover model_types {sorted(covered)}, "
            f"expected the union to be exactly {sorted(_MODEL_TYPES)} — one panel "
            "per model type, whether via separate calls or one combined call"
        )

    def test_contingency_calls_cover_both_rnn_and_lstm(self):
        tree = _parse()
        constants = _module_constants(tree)
        _, contingency_calls = self._cwc_calls()
        covered: set[str] = set()
        for call in contingency_calls:
            model_types = _resolve(_arg_value(call, 1, "model_types"), constants)
            if model_types:
                covered |= set(model_types)
        assert covered == _MODEL_TYPES, (
            f"the model_cwc_by_contingency call(s) cover model_types "
            f"{sorted(covered)}, expected the union to be exactly "
            f"{sorted(_MODEL_TYPES)}"
        )

    def test_an_explicit_seed_constant_is_defined(self):
        constants = _module_constants(_parse())
        seed_names = [
            name
            for name, value in constants.items()
            if "SEED" in name and _is_int_literal(value)
        ]
        assert seed_names, (
            "expected a module-level integer constant with 'SEED' in its "
            "name (e.g. `CWC_SEED = 0`), passed to the model CWC transforms "
            "so sampling is reproducible across runs"
        )

    def test_model_cwc_calls_pass_the_seed_constant_by_reference(self):
        """Every call's `seed` must reference the module-level SEED constant.

        Stronger than "resolves to some literal int": a bare inline `seed=0`
        at each call site would pass a looser check but let the four calls
        drift out of sync if only one call site is ever updated — the whole
        point of "a module-level seed constant passed to the model
        transforms" is a single source of truth.
        """
        tree = _parse()
        constants = _module_constants(tree)
        seed_names = {
            name for name, value in constants.items() if "SEED" in name and _is_int_literal(value)
        }
        hazard_calls, contingency_calls = self._cwc_calls()
        offenders = []
        for call in (*hazard_calls, *contingency_calls):
            seed_node = _arg_value(call, 3, "seed")
            if not (isinstance(seed_node, ast.Name) and seed_node.id in seed_names):
                offenders.append(call.lineno)
        assert not offenders, (
            f"call(s) at line(s) {offenders} do not pass `seed` as a "
            "reference to the module-level SEED constant (a bare inline "
            "literal, e.g. `seed=0`, is not enough — every call must share "
            "one named constant)"
        )

    def test_model_cwc_num_participants_is_not_a_hardcoded_literal(self):
        """``num_participants`` must come from the participant-count artifact.

        Directly pins the integration note: the model panels' pool size
        tracks the actual surviving human cohort, so a literal (even one that
        happens to equal today's count) would silently go stale the next time
        the participant pipeline reruns with a different final_n.
        """
        tree = _parse()
        constants = _module_constants(tree)
        hazard_calls, contingency_calls = self._cwc_calls()
        offenders = []
        for call in (*hazard_calls, *contingency_calls):
            node = _arg_value(call, 2, "num_participants")
            if node is None:
                offenders.append((call.lineno, "missing"))
                continue
            value = _resolve(node, constants)
            if _is_int_literal(value):
                offenders.append((call.lineno, value))
        assert not offenders, (
            f"num_participants resolves to a hardcoded literal at "
            f"{offenders} — it must be read from the participant counts "
            "artifact (data/cache/participants/participant_counts.json's "
            "'final_n'), not a fixed integer"
        )

    def test_source_references_the_participant_counts_artifact(self):
        source = _source()
        assert any(
            marker in source for marker in ("participant_counts.json", "ARTIFACT_COUNTS", "final_n")
        ), (
            "expected the script to reference the participant counts "
            "artifact (by filename, the participants.ARTIFACT_COUNTS "
            "constant, or its 'final_n' key) as the source of "
            "num_participants"
        )


# --------------------------------------------------------------------------- #
# Tier 1c — participant-side CWC call contracts
# --------------------------------------------------------------------------- #
class TestParticipantCwcCallContracts:
    def _calls(self) -> tuple[list[ast.Call], list[ast.Call]]:
        tree = _parse()
        return (
            _calls_to(tree, "participant_cwc_by_hazard"),
            _calls_to(tree, "participant_cwc_by_contingency"),
        )

    def test_participant_cwc_by_hazard_called_exactly_once(self):
        hazard_calls, _ = self._calls()
        assert len(hazard_calls) == 1, (
            f"expected exactly one call to participant_cwc_by_hazard (grouping "
            f"every participant against the same full-metadata buckets in one "
            f"pass); found {len(hazard_calls)}"
        )

    def test_participant_cwc_by_contingency_called_exactly_once(self):
        _, contingency_calls = self._calls()
        assert len(contingency_calls) == 1, (
            f"expected exactly one call to participant_cwc_by_contingency; "
            f"found {len(contingency_calls)}"
        )

    def test_participant_calls_are_not_inside_a_loop(self):
        tree = _parse()
        parents = _build_parent_map(tree)
        hazard_calls, contingency_calls = self._calls()
        offenders = [
            call.lineno
            for call in (*hazard_calls, *contingency_calls)
            if _has_loop_ancestor(call, parents)
        ]
        assert not offenders, (
            f"call(s) at line(s) {offenders} to participant_cwc_by_hazard/"
            "participant_cwc_by_contingency sit inside a loop or "
            "comprehension — these must run once against the full "
            "participant dict and full trial metadata, not once per "
            "participant"
        )

    def test_second_argument_is_a_full_metadata_frame_not_an_inline_subset(self):
        """The ``df_straight``/``df_bounce`` argument must be a bound name.

        Rules out the anti-pattern of slicing metadata inline per call (e.g.
        a per-participant filter expression) — the full straight/bounce
        metadata is built once and passed by reference.
        """
        hazard_calls, contingency_calls = self._calls()
        offenders = []
        for call in (*hazard_calls, *contingency_calls):
            metadata_node = _arg_value(call, 1, "df_straight") or _arg_value(
                call, 1, "df_bounce"
            )
            if not isinstance(metadata_node, ast.Name):
                offenders.append(call.lineno)
        assert not offenders, (
            f"call(s) at line(s) {offenders} pass a metadata argument that is "
            "not a simple bound name — the full straight/bounce trial "
            "metadata should be built once and referenced, not "
            "sliced/filtered inline at the call site"
        )

    def test_participant_cwc_dict_is_rebuilt_via_groupby_on_participant_id(self):
        """Positive check for the dict-rebuild step the transform needs.

        ``participant_cwc_by_hazard``/``by_contingency`` take a
        ``{participant_id: responses}`` mapping, not the flat
        ``participant_cwc.parquet`` table — the script must group the parquet
        by 'Participant ID' to build it.
        """
        tree = _parse()
        constants = _module_constants(tree)
        groupby_calls = _calls_to(tree, "groupby")
        resolved = [
            _resolve(call.args[0] if call.args else _kwarg_nodes(call).get("by"), constants)
            for call in groupby_calls
        ]
        assert "Participant ID" in resolved, (
            "expected a `.groupby('Participant ID')` call rebuilding the "
            "participant_cwc_dict from the participant_cwc.parquet artifact"
        )

    def test_no_manual_groupby_on_hazard_or_contingency_columns(self):
        """The script must delegate condition-grouping to the participants module.

        A `.groupby('Hazard Rate')` / `.groupby('Contingency')` in the script
        itself would mean the hazard/contingency aggregation was
        reimplemented locally instead of going through
        participant_cwc_by_hazard/by_contingency.
        """
        tree = _parse()
        constants = _module_constants(tree)
        groupby_calls = _calls_to(tree, "groupby")
        forbidden = {"Hazard Rate", "Contingency"}
        offenders = []
        for call in groupby_calls:
            by_node = call.args[0] if call.args else _kwarg_nodes(call).get("by")
            value = _resolve(by_node, constants)
            keys = {value} if isinstance(value, str) else set(value) if isinstance(value, (list, tuple)) else set()
            if keys & forbidden:
                offenders.append((call.lineno, value))
        assert not offenders, (
            f"found direct `.groupby(...)` call(s) on hazard/contingency "
            f"columns at {offenders} — condition aggregation must go through "
            "participant_cwc_by_hazard/participant_cwc_by_contingency, not a "
            "raw groupby on the parquet"
        )


# --------------------------------------------------------------------------- #
# Tier 2 — deck-grammar content contracts (real cached data, no script needed)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not _HAS_PARTICIPANT_MODEL_DATA,
    reason=f"real {_PARTICIPANT_DATASET} model-state artifacts not found under {MODEL_STATES_DIR}",
)
@pytest.mark.skipif(_NUM_PARTICIPANTS is None, reason="participant_counts.json not found")
class TestModelContentContracts:
    """Runs the real transforms with the cohort size from the counts artifact."""

    @pytest.fixture(scope="class")
    def hazard_rnn(self):
        return transforms.model_cwc_by_hazard(
            dataset=_PARTICIPANT_DATASET,
            model_types=("rnn",),
            num_participants=_NUM_PARTICIPANTS,
            seed=0,
        )

    @pytest.fixture(scope="class")
    def hazard_lstm(self):
        return transforms.model_cwc_by_hazard(
            dataset=_PARTICIPANT_DATASET,
            model_types=("lstm",),
            num_participants=_NUM_PARTICIPANTS,
            seed=0,
        )

    @pytest.fixture(scope="class")
    def contingency_rnn(self):
        return transforms.model_cwc_by_contingency(
            dataset=_PARTICIPANT_DATASET,
            model_types=("rnn",),
            num_participants=_NUM_PARTICIPANTS,
            seed=0,
        )

    @pytest.fixture(scope="class")
    def contingency_lstm(self):
        return transforms.model_cwc_by_contingency(
            dataset=_PARTICIPANT_DATASET,
            model_types=("lstm",),
            num_participants=_NUM_PARTICIPANTS,
            seed=0,
        )

    def test_hazard_rnn_has_two_levels_and_three_grayzone_positions(self, hazard_rnn):
        assert set(hazard_rnn["Hazard Rate"].astype(str).unique()) == {"Low", "High"}
        assert hazard_rnn["Grayzone Position"].nunique() == 3

    def test_hazard_lstm_has_two_levels_and_three_grayzone_positions(self, hazard_lstm):
        assert set(hazard_lstm["Hazard Rate"].astype(str).unique()) == {"Low", "High"}
        assert hazard_lstm["Grayzone Position"].nunique() == 3

    def test_contingency_rnn_has_three_levels(self, contingency_rnn):
        assert set(contingency_rnn["Contingency"].astype(str).unique()) == {
            "Low",
            "Medium",
            "High",
        }

    def test_contingency_lstm_has_three_levels(self, contingency_lstm):
        assert set(contingency_lstm["Contingency"].astype(str).unique()) == {
            "Low",
            "Medium",
            "High",
        }


@pytest.mark.skipif(
    not _HAS_PARTICIPANTS_ARTIFACTS,
    reason=f"participant artifacts not found under {PARTICIPANTS_DIR}",
)
@pytest.mark.skipif(
    not _HAS_PARTICIPANT_MODEL_DATA,
    reason=f"real {_PARTICIPANT_DATASET} trial metadata not found under {MODEL_STATES_DIR}",
)
class TestParticipantContentContracts:
    """Runs the real participants transforms against the landed artifacts."""

    @pytest.fixture(scope="class")
    def participant_cwc_dict(self):
        df = pd.read_parquet(PARTICIPANTS_DIR / participants.ARTIFACT_CWC)
        return {pid: group.set_index("Video ID") for pid, group in df.groupby("Participant ID")}

    @pytest.fixture(scope="class")
    def df_straight(self):
        df_meta = transforms.trial_metadata(_PARTICIPANT_DATASET)
        return df_meta[df_meta["trial"] == "Straight"]

    @pytest.fixture(scope="class")
    def df_bounce(self):
        df_meta = transforms.trial_metadata(_PARTICIPANT_DATASET)
        return df_meta[df_meta["trial"] == "Bounce"]

    def test_participant_hazard_frame_covers_the_whole_cohort(
        self, participant_cwc_dict, df_straight
    ):
        """The cohort size is cross-checked against the counts artifact, never
        against a literal: pinning today's ``final_n`` here would contradict the
        same rule this file imposes on the script (resolved ambiguity 2) and go
        stale the next time the participant pipeline reruns.
        """
        df_hazard = participants.participant_cwc_by_hazard(participant_cwc_dict, df_straight)
        assert df_hazard["Participant ID"].nunique() == _NUM_PARTICIPANTS
        assert set(df_hazard["Hazard Rate"].dropna().astype(str).unique()) == {"Low", "High"}
        assert df_hazard["Grayzone Position"].dropna().nunique() == 3

    def test_participant_contingency_frame_covers_the_whole_cohort(
        self, participant_cwc_dict, df_bounce
    ):
        df_contingency = participants.participant_cwc_by_contingency(
            participant_cwc_dict, df_bounce
        )
        assert df_contingency["Participant ID"].nunique() == _NUM_PARTICIPANTS
        assert set(df_contingency["Contingency"].dropna().astype(str).unique()) == {
            "Low",
            "Medium",
            "High",
        }


# --------------------------------------------------------------------------- #
# Tier 3 — style contracts
# --------------------------------------------------------------------------- #
class TestStyleContracts:
    def _classified_swarm_calls(self):
        tree = _parse()
        constants = _module_constants(tree)
        classified = []
        for call in _calls_to(tree, "plot_cwc_swarm"):
            x_val = _resolve(_arg_value(call, 1, "x"), constants)
            hue_node = _arg_value(call, 3, "hue")
            hue_val = _resolve(hue_node, constants) if hue_node is not None else None
            legend_node = _kwarg_nodes(call).get("legend")
            legend_val = _resolve(legend_node, constants) if legend_node is not None else True
            classified.append(
                {"call": call, "x": x_val, "hue": hue_val, "legend": legend_val}
            )
        return classified

    def test_top_row_panels_split_by_hazard_rate_with_a_legend_on_every_panel(self):
        hazard_panels = [c for c in self._classified_swarm_calls() if c["x"] == "Grayzone Position"]
        assert len(hazard_panels) == 3, (
            f"expected exactly 3 plot_cwc_swarm calls with x='Grayzone Position' "
            f"(one per straight-trial panel: participants, RNN, LSTM); found "
            f"{len(hazard_panels)}"
        )
        offenders = [c["call"].lineno for c in hazard_panels if c["hue"] != "Hazard Rate"]
        assert not offenders, (
            f"call(s) at line(s) {offenders} do not split on hue='Hazard Rate'"
        )
        no_legend = [c["call"].lineno for c in hazard_panels if c["legend"] is False]
        assert not no_legend, (
            f"call(s) at line(s) {no_legend} suppress the legend — every "
            "straight-trial panel is self-contained and must show its own "
            "Hazard Rate legend, including the participants panel"
        )

    def test_bottom_row_panels_are_single_family_with_no_legend(self):
        contingency_panels = [
            c for c in self._classified_swarm_calls() if c["x"] == "Contingency"
        ]
        assert len(contingency_panels) == 3, (
            f"expected exactly 3 plot_cwc_swarm calls with x='Contingency' "
            f"(one per bounce-trial panel: participants, RNN, LSTM); found "
            f"{len(contingency_panels)}"
        )
        offenders = [
            c["call"].lineno
            for c in contingency_panels
            if not (c["hue"] is None or c["hue"] == c["x"])
        ]
        assert not offenders, (
            f"call(s) at line(s) {offenders} pass a hue other than None/x on "
            "the bounce row — cwc_plots.plot_cwc_swarm only omits the legend "
            "in family mode (hue is None or hue == x)"
        )

    def test_dashed_gridlines_are_enabled_somewhere_in_the_render_path(self):
        tree = _parse()
        constants = _module_constants(tree)
        grid_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "grid"
        ]
        assert grid_calls, (
            "expected at least one `<ax>.grid(...)` (or equivalent) call "
            "enabling gridlines somewhere in the script's render cells"
        )
        dashed_specs = {"--", ":", "-.", "dashed", "dashdot", "dotted"}
        has_dashed = False
        for call in grid_calls:
            kwargs = _kwarg_nodes(call)
            linestyle_node = kwargs.get("linestyle") or kwargs.get("ls")
            if linestyle_node is None:
                continue
            value = _resolve(linestyle_node, constants)
            if isinstance(value, str) and value in dashed_specs:
                has_dashed = True
                break
        assert has_dashed, (
            "expected at least one `.grid(..., linestyle=...)` call using a "
            "dashed linestyle ('--', ':', '-.', 'dashed', etc.) — the "
            "gridlines must actually be dashed, not merely enabled"
        )


# --------------------------------------------------------------------------- #
# Tier 4 — headless run
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fig3_run(tmp_path_factory):
    """Run figures/fig3_task_results.py headlessly, once per test session.

    Mirrors tests/test_fig2_panels.py's / tests/test_fig5_panels.py's fixture:
    the tier-2 transform cache is redirected to an isolated tmp dir via
    ``LIC_FIG_CACHE_DIR`` so this never depends on, or pollutes,
    ``data/cache/fig_transforms``. Panels, by contrast, land in the real
    ``figures/panels/fig3/`` — that is the actual deliverable.
    """
    cache_dir = tmp_path_factory.mktemp("fig_transforms_cache")
    env = dict(os.environ)
    env["LIC_FIG_CACHE_DIR"] = str(cache_dir)

    result = subprocess.run(
        [sys.executable, str(FIG3_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result, cache_dir, env


_requires_real_data = pytest.mark.skipif(
    not (_HAS_PARTICIPANT_MODEL_DATA and _HAS_PARTICIPANTS_ARTIFACTS),
    reason=(
        f"real {_PARTICIPANT_DATASET} model-state artifacts and/or participant "
        f"artifacts not found under {MODEL_STATES_DIR} / {PARTICIPANTS_DIR}"
    ),
)


@pytest.mark.slow
@pytest.mark.integration
@_requires_real_data
class TestHeadlessRun:
    def test_script_exists(self):
        assert FIG3_SCRIPT.exists(), f"figure script not found: {FIG3_SCRIPT}"

    def test_exits_zero(self, fig3_run):
        result, _cache_dir, _env = fig3_run
        assert result.returncode == 0, (
            "figures/fig3_task_results.py exited "
            f"{result.returncode}\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )


@pytest.mark.slow
@pytest.mark.integration
@_requires_real_data
class TestPanelOutputs:
    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_writes_exactly_the_expected_panel(self, fig3_run, panel_name):
        result, _cache_dir, _env = fig3_run
        if result.returncode != 0:
            pytest.fail(f"headless run failed:\n{result.stderr}")
        panel_path = PANELS_DIR / panel_name
        assert panel_path.exists(), f"expected panel not written: {panel_path}"

    def test_writes_no_unexpected_panels(self, fig3_run):
        result, _cache_dir, _env = fig3_run
        if result.returncode != 0:
            pytest.fail(f"headless run failed:\n{result.stderr}")
        if not PANELS_DIR.exists():
            pytest.fail(f"panels directory was never created: {PANELS_DIR}")
        written = {p.name for p in PANELS_DIR.glob("*.svg")}
        unexpected = written - set(EXPECTED_PANELS)
        assert not unexpected, f"unexpected panel(s) written: {sorted(unexpected)}"

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_has_live_text_elements(self, fig3_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_exactly_the_expected_panel")
        content = panel_path.read_text()
        assert "<text" in content, f"{panel_name} has no live <text> elements"

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_is_a_single_self_contained_svg(self, fig3_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_exactly_the_expected_panel")
        content = panel_path.read_text()
        assert content.count("<svg") == 1, (
            f"{panel_name} does not look like a single self-contained panel "
            f"(found {content.count('<svg')} <svg> tags)"
        )


@pytest.mark.slow
@pytest.mark.integration
@_requires_real_data
class TestDeterminism:
    def test_second_run_is_byte_identical(self, fig3_run, tmp_path_factory):
        result, _cache_dir, env = fig3_run
        if result.returncode != 0:
            pytest.fail(f"first headless run failed:\n{result.stderr}")

        first_bytes = {}
        for name in EXPECTED_PANELS:
            path = PANELS_DIR / name
            if not path.exists():
                pytest.fail(f"expected panel not written on first run: {path}")
            first_bytes[name] = path.read_bytes()

        # A fresh, separate transform cache dir for the second run: reusing
        # the first run's warm cache would let a stale (or entirely unseeded)
        # CWC sample get served from disk, making the byte-identity check
        # pass without the seed actually doing anything.
        second_cache_dir = tmp_path_factory.mktemp("fig_transforms_cache_second")
        second_env = dict(env)
        second_env["LIC_FIG_CACHE_DIR"] = str(second_cache_dir)

        second = subprocess.run(
            [sys.executable, str(FIG3_SCRIPT)],
            cwd=REPO_ROOT,
            env=second_env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert second.returncode == 0, (
            "second headless run of figures/fig3_task_results.py exited "
            f"{second.returncode}\n--- stdout ---\n{second.stdout}\n"
            f"--- stderr ---\n{second.stderr}"
        )

        mismatched = []
        for name in EXPECTED_PANELS:
            path = PANELS_DIR / name
            if not path.exists():
                mismatched.append(f"{name} (missing on second run)")
                continue
            if path.read_bytes() != first_bytes[name]:
                mismatched.append(name)
        assert not mismatched, (
            f"panel(s) not byte-identical across two headless runs: {mismatched} "
            "-- check that every source of randomness (the model CWC "
            "transforms' sampling in particular) is seeded by an explicit "
            "constant"
        )

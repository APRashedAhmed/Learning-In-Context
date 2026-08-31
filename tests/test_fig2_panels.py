"""Contract tests for ``figures/fig2_ideal_observer.py`` (marimo, dual-use).

Figure 2 renders four panels under ``figures/panels/fig2/``:

* ``estimate_curve_hazard_rate.svg`` / ``estimate_curve_contingency.svg`` —
  line plots of the ideal Bayesian observer's belief curves, sourced from
  ``transforms.ideal_observer_belief_curves`` (see
  ``tests/test_ideal_observer_curves.py`` for that transform's own contract).
  The hazard variant sweeps ``pccnvc`` over the transform's default two
  levels; the contingency variant sweeps ``pccovc`` over three levels on
  wall-bounce trials (``trial_type="Bounce"``).
* ``cwc_hazard_rate.svg`` / ``cwc_contingency.svg`` — ideal-observer-only
  swarm-plus-mean panels drawn by ``cwc_plots.plot_cwc_swarm``, fed by
  ``transforms.model_cwc_by_hazard`` / ``transforms.model_cwc_by_contingency``
  restricted to ``model_types=("ibo",)``.

This file pins three tiers:

1. Static script structure (AST-level, no data, no subprocess) — the script
   exists, is a marimo app, imports the right modules, avoids computing
   inside the script what belongs in ``transforms``, names all four panels,
   and does not import the sibling ``hmdcpd-analysis`` repo or define its own
   observer class.
2. Content contracts for the two transform families, checked against the
   script's own call sites via AST (and, where real cached data is present,
   against the transform's actual output) rather than against rendered SVGs.
3. A headless-run tier (``python figures/fig2_ideal_observer.py``) that lands
   exactly the four expected SVGs — freshly written by that run, not left over
   from an earlier one — each carrying live text and real path data, and
   reproduces byte-identical output on a second run.

The CWC dataset is deliberately left to a single named constant
(``CWC_DATASET``) rather than pinned to one string: whether the CWC swarms
draw from the participant-scale pool or the same control pool the estimate
curves use is an open call the implementation and a visual check settle
together, so this file only requires the constant to exist, be used
consistently, and hold one of the two candidate dataset names.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

try:
    from learning_in_context.visualization import transforms

    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - only if transforms.py itself breaks
    transforms = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG2_SCRIPT = REPO_ROOT / "figures" / "fig2_ideal_observer.py"
PANELS_DIR = REPO_ROOT / "figures" / "panels" / "fig2"
MODEL_STATES_DIR = REPO_ROOT / "data" / "cache" / "model_states"

PANEL_BASENAMES = [
    "estimate_curve_hazard_rate",
    "estimate_curve_contingency",
    "cwc_hazard_rate",
    "cwc_contingency",
]
EXPECTED_PANELS = [f"{name}.svg" for name in PANEL_BASENAMES]

CWC_DATASET_CHOICES = {"participant_dataset", "control_dataset"}

_PATH_D_RE = re.compile(r'<path\b[^>]*\sd="([^"]*)"')

# Length, in characters of SVG path data, above which a ``<path>`` is carrying
# real content rather than axes furniture. The four panels' longest paths run
# from ~440 to ~880 characters; their spines, ticks and axes patches all stay
# under 100, so the floor sits clear of both.
_SUBSTANTIAL_PATH_CHARS = 150


def _dataset_ready(name: str) -> bool:
    d = MODEL_STATES_DIR / name
    return (d / "trial_meta.csv").exists() and (d / "samples.npy").exists()


_HAS_CONTROL_DATA = _dataset_ready("control_dataset")
_HAS_ALL_CWC_CANDIDATE_DATA = all(_dataset_ready(name) for name in CWC_DATASET_CHOICES)


# --------------------------------------------------------------------------- #
# AST plumbing
# --------------------------------------------------------------------------- #
def _parse() -> ast.Module:
    return ast.parse(FIG2_SCRIPT.read_text(), filename=str(FIG2_SCRIPT))


def _source() -> str:
    return FIG2_SCRIPT.read_text()


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
    """Resolve a call-keyword AST node to a literal Python value, or ``None``.

    Direct literals resolve via ``ast.literal_eval``. A bare ``Name`` resolves
    through a module-level ``NAME = <literal>`` constant assignment found
    anywhere in the script, so a script may equally well pass a literal
    inline or thread a named constant through.
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


def _imports_name(tree: ast.Module, name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound = alias.asname or alias.name
                if bound == name and (
                    node.module is not None and "visualization" in node.module
                ):
                    return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = (alias.asname or alias.name).rsplit(".", 1)[-1]
                if bound == name and name in alias.name:
                    return True
    return False


class Sig:
    """Lazily-fetched defaults from the real transform signatures.

    Failing softly (via ``pytest.skip``/``pytest.fail`` at call sites, not
    here) keeps this file importable even before ``transforms`` gains the
    fig2-only pieces this contract does not own.
    """

    @staticmethod
    def default(func_name: str, param_name: str):
        assert transforms is not None, _IMPORT_ERROR
        fn = getattr(transforms, func_name)
        return inspect.signature(fn).parameters[param_name].default


# --------------------------------------------------------------------------- #
# Tier 1 — static script structure
# --------------------------------------------------------------------------- #
class TestScriptExists:
    def test_script_exists(self):
        assert FIG2_SCRIPT.exists(), f"figure script not found: {FIG2_SCRIPT}"


class TestScriptStructure:
    def test_defines_a_marimo_app(self):
        """`app` must be built by a call to ``.App(...)``, not merely assigned.

        A bare ``app = something`` satisfies the letter of "assigns app" while
        leaving the file unrunnable as a notebook, so the assigned value is
        checked to be a ``<module>.App(...)`` call.
        """
        tree = _parse()
        source = _source()
        assert "import marimo" in source
        assigns_app = any(
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "app" for t in node.targets)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "App"
            for node in ast.walk(tree)
        )
        assert assigns_app, "expected a module-level `app = marimo.App(...)` assignment"

    def test_imports_transforms_module(self):
        assert _imports_name(_parse(), "transforms")

    def test_imports_cwc_plots_module(self):
        assert _imports_name(_parse(), "cwc_plots")

    def test_imports_paper_style_module(self):
        assert _imports_name(_parse(), "paper_style")

    def test_defines_save_svgs_switch(self):
        tree = _parse()
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(t, ast.Name) and t.id == "save_svgs" for t in node.targets
            ):
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

        A figure script reads cached artifacts only through memoized
        transforms; catching a direct load here is the cheapest way to guard
        against a panel silently growing its own copy of tier-2 logic.
        """
        tree = _parse()
        offenders = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ):
                continue
            root = node.func.value
            root_name = root.id if isinstance(root, ast.Name) else None
            if root_name in {"np", "numpy"} and node.func.attr == "load":
                offenders.append(node.lineno)
            if root_name in {"pd", "pandas"} and node.func.attr == "read_csv":
                offenders.append(node.lineno)
        assert not offenders, (
            f"figures/fig2_ideal_observer.py loads cache arrays directly at "
            f"line(s) {offenders}; route this through "
            "learning_in_context.visualization.transforms instead"
        )

    def test_save_panel_calls_name_exactly_the_four_expected_panels(self):
        """Checks the actual ``save_panel`` call arguments, not prose.

        A docstring can list every panel name without a single ``save_panel``
        call actually using it (a typo'd name, a missing panel) — so this
        resolves the third argument (positional or ``name=``) of every
        ``save_panel(...)`` call in the script and compares that set against
        the four expected panel names, rather than grepping the source text.
        """
        tree = _parse()
        constants = _module_constants(tree)
        calls = _calls_to(tree, "save_panel")
        assert calls, "expected at least one call to paper_style.save_panel(...)"

        names: set[str] = set()
        unresolved = []
        for call in calls:
            name_node = None
            if len(call.args) >= 3:
                name_node = call.args[2]
            else:
                kwargs = _kwarg_nodes(call)
                name_node = kwargs.get("name")
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

    def test_save_panel_calls_target_figure_two(self):
        """The panels must land under ``fig2/``, not another figure's directory.

        ``save_panel``'s second argument picks the output directory, so a
        correctly named panel written with the wrong figure number would still
        satisfy the name check above while landing in, say,
        ``figures/panels/fig5/``.
        """
        tree = _parse()
        constants = _module_constants(tree)
        calls = _calls_to(tree, "save_panel")
        assert calls, "expected at least one call to paper_style.save_panel(...)"

        wrong = []
        for call in calls:
            if len(call.args) >= 2:
                fig_no_node = call.args[1]
            else:
                fig_no_node = _kwarg_nodes(call).get("fig_no")
            value = _resolve(fig_no_node, constants)
            if value not in (2, "2"):
                wrong.append((call.lineno, value))
        assert not wrong, (
            f"save_panel call(s) at line(s) {[line for line, _ in wrong]} target "
            f"figure {[value for _, value in wrong]}, expected figure 2"
        )

    def test_plot_cwc_swarm_is_actually_called(self):
        calls = _calls_to(_parse(), "plot_cwc_swarm")
        assert len(calls) >= 2, (
            f"expected at least 2 calls to cwc_plots.plot_cwc_swarm (one per "
            f"CWC panel); found {len(calls)} — the CWC panels must go "
            "through the landed renderer, not a hand-rolled swarm"
        )

    def test_does_not_import_the_sibling_analysis_repo(self):
        assert "hmdcpd" not in _source()

    def test_defines_no_observer_class_of_its_own(self):
        tree = _parse()
        observer_classes = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and "Observer" in node.name
        ]
        assert not observer_classes, (
            f"figures/fig2_ideal_observer.py defines its own observer-like "
            f"class(es): {observer_classes}; belief propagation must go "
            "through the imported model via transforms.ideal_observer_belief_curves"
        )


# --------------------------------------------------------------------------- #
# Tier 2a — estimate-curve panel content contracts
# --------------------------------------------------------------------------- #
class TestEstimateCurveCallContracts:
    """AST-level pins on the two calls to ``ideal_observer_belief_curves``."""

    def _calls(self) -> list[ast.Call]:
        return _calls_to(_parse(), "ideal_observer_belief_curves")

    def test_exactly_two_calls_to_the_belief_curve_transform(self):
        calls = self._calls()
        assert len(calls) == 2, (
            f"expected exactly 2 calls to ideal_observer_belief_curves (one "
            f"hazard variant, one contingency variant); found {len(calls)}"
        )

    def _resolved(self, call: ast.Call, constants: dict[str, object]) -> dict[str, object]:
        kwargs = _kwarg_nodes(call)
        sweep = (
            _resolve(kwargs["sweep"], constants)
            if "sweep" in kwargs
            else Sig.default("ideal_observer_belief_curves", "sweep")
        )
        trial_type = (
            _resolve(kwargs["trial_type"], constants)
            if "trial_type" in kwargs
            else Sig.default("ideal_observer_belief_curves", "trial_type")
        )
        levels = (
            _resolve(kwargs["levels"], constants)
            if "levels" in kwargs
            else Sig.default("ideal_observer_belief_curves", "levels")
        )
        return {"sweep": sweep, "trial_type": trial_type, "levels": levels}

    def _hazard_and_contingency_calls(self):
        tree = _parse()
        constants = _module_constants(tree)
        calls = _calls_to(tree, "ideal_observer_belief_curves")
        resolved = [self._resolved(c, constants) for c in calls]
        hazard = [r for r in resolved if r["sweep"] == "hazard"]
        contingency = [r for r in resolved if r["sweep"] == "contingency"]
        assert len(hazard) == 1, (
            f"expected exactly one call with sweep='hazard' (explicit or "
            f"defaulted); resolved sweeps were {[r['sweep'] for r in resolved]}"
        )
        assert len(contingency) == 1, (
            f"expected exactly one call with sweep='contingency'; resolved "
            f"sweeps were {[r['sweep'] for r in resolved]}"
        )
        return hazard[0], contingency[0]

    def test_hazard_variant_uses_the_transforms_default_two_level_sweep(self):
        hazard, _contingency = self._hazard_and_contingency_calls()
        levels = hazard["levels"]
        assert levels is not None, (
            "the hazard call's `levels` argument could not be resolved to a "
            "literal or a module-level constant"
        )
        assert len(levels) == 2, f"expected 2 hazard levels, got {levels!r}"

    def test_contingency_variant_supplies_three_levels(self):
        _hazard, contingency = self._hazard_and_contingency_calls()
        levels = contingency["levels"]
        assert levels is not None, (
            "the contingency call's `levels` argument could not be resolved to "
            "a literal or a module-level constant"
        )
        assert len(levels) == 3, f"expected 3 contingency levels, got {levels!r}"

    def test_contingency_variant_passes_trial_type_bounce(self):
        _hazard, contingency = self._hazard_and_contingency_calls()
        assert contingency["trial_type"] == "Bounce", (
            f"expected the contingency variant to pass trial_type='Bounce' "
            f"(the transform's default is 'Straight', so this must be "
            f"explicit); resolved trial_type={contingency['trial_type']!r}"
        )


@pytest.mark.skipif(
    not _HAS_CONTROL_DATA,
    reason=f"real control_dataset artifacts not found under {MODEL_STATES_DIR / 'control_dataset'}",
)
class TestEstimateCurveTransformBehavior:
    """Runs the actual transform with the script's own resolved call args.

    Cheap relative to the headless-run tier: one belief-propagation pass per
    swept level over a single exemplar trial, not a full script render.
    """

    def _resolved_kwargs(self, sweep: str) -> dict[str, object]:
        tree = _parse()
        constants = _module_constants(tree)
        calls = _calls_to(tree, "ideal_observer_belief_curves")
        for call in calls:
            kwargs = _kwarg_nodes(call)
            resolved_sweep = (
                _resolve(kwargs["sweep"], constants)
                if "sweep" in kwargs
                else Sig.default("ideal_observer_belief_curves", "sweep")
            )
            if resolved_sweep != sweep:
                continue
            out: dict[str, object] = {}
            for name in ("dataset", "trial_type", "sweep", "levels", "idx_time"):
                if name in kwargs:
                    value = _resolve(kwargs[name], constants)
                    if value is not None:
                        out[name] = value
            out.setdefault("sweep", sweep)
            return out
        pytest.fail(f"no call to ideal_observer_belief_curves resolves to sweep={sweep!r}")

    def test_hazard_panel_transform_output_has_two_levels(self):
        kwargs = self._resolved_kwargs("hazard")
        df = transforms.ideal_observer_belief_curves(**kwargs)
        assert df["level"].nunique() == 2

    def test_contingency_panel_transform_output_has_three_levels(self):
        kwargs = self._resolved_kwargs("contingency")
        df = transforms.ideal_observer_belief_curves(**kwargs)
        assert df["level"].nunique() == 3


# --------------------------------------------------------------------------- #
# Tier 2b — CWC panel content contracts
# --------------------------------------------------------------------------- #
class TestCwcCallContracts:
    def test_cwc_dataset_constant_is_defined_and_named_candidate(self):
        constants = _module_constants(_parse())
        assert "CWC_DATASET" in constants, (
            "expected a module-level `CWC_DATASET = ...` constant naming the "
            "dataset the CWC panels read from"
        )
        assert constants["CWC_DATASET"] in CWC_DATASET_CHOICES, (
            f"CWC_DATASET={constants['CWC_DATASET']!r} is not one of the "
            f"candidate dataset names {sorted(CWC_DATASET_CHOICES)}"
        )

    def _cwc_calls(self) -> tuple[list[ast.Call], list[ast.Call]]:
        tree = _parse()
        return (
            _calls_to(tree, "model_cwc_by_hazard"),
            _calls_to(tree, "model_cwc_by_contingency"),
        )

    def test_exactly_one_call_each_to_the_cwc_transforms(self):
        hazard_calls, contingency_calls = self._cwc_calls()
        assert len(hazard_calls) == 1, (
            f"expected exactly one call to model_cwc_by_hazard; found "
            f"{len(hazard_calls)}"
        )
        assert len(contingency_calls) == 1, (
            f"expected exactly one call to model_cwc_by_contingency; found "
            f"{len(contingency_calls)}"
        )

    def test_both_cwc_calls_reference_the_dataset_constant_not_a_literal(self):
        hazard_calls, contingency_calls = self._cwc_calls()
        for call in (*hazard_calls, *contingency_calls):
            kwargs = _kwarg_nodes(call)
            dataset_node = kwargs.get("dataset")
            assert dataset_node is not None, (
                f"line {call.lineno}: call to "
                f"{call.func.attr} does not pass a `dataset` keyword"
            )
            assert isinstance(dataset_node, ast.Name) and dataset_node.id == "CWC_DATASET", (
                f"line {call.lineno}: `dataset` is not a reference to the "
                "CWC_DATASET constant — the dataset name must not be "
                "hardcoded independently in each CWC call"
            )

    def test_both_cwc_calls_are_ibo_only(self):
        tree = _parse()
        constants = _module_constants(tree)
        hazard_calls, contingency_calls = self._cwc_calls()
        for call in (*hazard_calls, *contingency_calls):
            kwargs = _kwarg_nodes(call)
            model_types = _resolve(kwargs.get("model_types"), constants)
            assert model_types == ("ibo",), (
                f"line {call.lineno}: expected model_types=('ibo',), resolved "
                f"{model_types!r}"
            )

    def test_an_explicit_seed_constant_is_defined(self):
        constants = _module_constants(_parse())
        seed_names = [
            name for name, value in constants.items()
            if "SEED" in name and isinstance(value, int) and not isinstance(value, bool)
        ]
        assert seed_names, (
            "expected a module-level integer constant with 'SEED' in its "
            "name (e.g. `CWC_SEED = 0`), passed to the CWC transforms so "
            "sampling is reproducible across runs"
        )

    def test_both_cwc_calls_pass_a_resolvable_integer_seed(self):
        tree = _parse()
        constants = _module_constants(tree)
        hazard_calls, contingency_calls = self._cwc_calls()
        for call in (*hazard_calls, *contingency_calls):
            kwargs = _kwarg_nodes(call)
            seed_node = kwargs.get("seed")
            assert seed_node is not None, (
                f"line {call.lineno}: call to {call.func.attr} does not pass "
                "a `seed` keyword"
            )
            seed_value = _resolve(seed_node, constants)
            assert isinstance(seed_value, int) and not isinstance(seed_value, bool), (
                f"line {call.lineno}: `seed` does not resolve to a literal "
                f"integer (resolved {seed_value!r}); it must be a fixed, "
                "reproducible value, not a runtime-computed one"
            )


# --------------------------------------------------------------------------- #
# Tier 3 — headless run
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fig2_run(tmp_path_factory):
    """Run figures/fig2_ideal_observer.py headlessly, once per test session.

    Mirrors tests/test_fig5_panels.py's fixture: the tier-2 transform cache
    is redirected to an isolated tmp dir via ``LIC_FIG_CACHE_DIR`` so this
    test never depends on, or pollutes, ``data/cache/fig_transforms``.
    Panels, by contrast, land in the real ``figures/panels/fig2/`` — that is
    the actual deliverable.

    The wall-clock instant just before the run is returned alongside the
    result: ``figures/panels/`` is regenerable but not cleaned between runs,
    so a panel left behind by an earlier run would otherwise satisfy every
    existence check even if this run wrote nothing at all.
    """
    cache_dir = tmp_path_factory.mktemp("fig_transforms_cache")
    env = dict(os.environ)
    env["LIC_FIG_CACHE_DIR"] = str(cache_dir)

    started = time.time()
    result = subprocess.run(
        [sys.executable, str(FIG2_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result, cache_dir, env, started


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_ALL_CWC_CANDIDATE_DATA,
    reason=(
        "real model-states artifacts not found under "
        f"{MODEL_STATES_DIR} for one or more of {sorted(CWC_DATASET_CHOICES)}"
    ),
)
class TestHeadlessRun:
    def test_script_exists(self):
        assert FIG2_SCRIPT.exists(), f"figure script not found: {FIG2_SCRIPT}"

    def test_exits_zero(self, fig2_run):
        result, _cache_dir, _env, _started = fig2_run
        assert result.returncode == 0, (
            "figures/fig2_ideal_observer.py exited "
            f"{result.returncode}\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_ALL_CWC_CANDIDATE_DATA,
    reason=(
        "real model-states artifacts not found under "
        f"{MODEL_STATES_DIR} for one or more of {sorted(CWC_DATASET_CHOICES)}"
    ),
)
class TestPanelOutputs:
    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_writes_exactly_the_expected_panel(self, fig2_run, panel_name):
        result, _cache_dir, _env, _started = fig2_run
        if result.returncode != 0:
            pytest.fail(f"headless run failed:\n{result.stderr}")
        panel_path = PANELS_DIR / panel_name
        assert panel_path.exists(), f"expected panel not written: {panel_path}"

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_was_rewritten_by_this_run(self, fig2_run, panel_name):
        """The panel on disk came from this run, not from an earlier one.

        ``figures/panels/`` is regenerable but never cleaned between runs, so
        existence alone cannot tell a freshly written panel from a leftover: a
        script that stopped exporting entirely — the save toggle defaulted off,
        say — would still pass every check in this class, and would pass the
        byte-identity check too, since two runs that write nothing leave the
        same stale bytes behind.
        """
        result, _cache_dir, _env, started = fig2_run
        if result.returncode != 0:
            pytest.fail(f"headless run failed:\n{result.stderr}")
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_exactly_the_expected_panel")
        # One second of slack absorbs coarse filesystem mtime granularity.
        assert panel_path.stat().st_mtime >= started - 1.0, (
            f"{panel_name} predates the headless run, so this run did not "
            "write it — the file on disk is a leftover from an earlier run"
        )

    def test_writes_no_unexpected_panels(self, fig2_run):
        result, _cache_dir, _env, _started = fig2_run
        if result.returncode != 0:
            pytest.fail(f"headless run failed:\n{result.stderr}")
        if not PANELS_DIR.exists():
            pytest.fail(f"panels directory was never created: {PANELS_DIR}")
        written = {p.name for p in PANELS_DIR.glob("*.svg")}
        unexpected = written - set(EXPECTED_PANELS)
        assert not unexpected, f"unexpected panel(s) written: {sorted(unexpected)}"

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_has_live_text_elements(self, fig2_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_exactly_the_expected_panel")
        content = panel_path.read_text()
        assert "<text" in content, f"{panel_name} has no live <text> elements"

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_has_nonempty_path_data(self, fig2_run, panel_name):
        panel_path = PANELS_DIR / panel_name
        if not panel_path.exists():
            pytest.skip("panel not written; see test_writes_exactly_the_expected_panel")
        content = panel_path.read_text()
        matches = _PATH_D_RE.findall(content)
        assert matches, f"{panel_name} has no <path d=...> elements"
        assert any(d.strip() for d in matches), f"{panel_name}'s <path> elements carry no data"
        # Spines, tick marks and the axes patch are all short paths, so an
        # empty axes satisfies the checks above. A drawn curve, or the marker
        # glyph a swarm reuses, is an order of magnitude longer — requiring one
        # such path is what separates a rendered panel from an empty frame.
        longest = max(len(d) for d in matches)
        assert longest >= _SUBSTANTIAL_PATH_CHARS, (
            f"{panel_name}'s longest <path d=...> is {longest} characters, "
            f"under the {_SUBSTANTIAL_PATH_CHARS} expected of a drawn curve or "
            "marker glyph — the panel looks like an empty set of axes"
        )

    @pytest.mark.parametrize("panel_name", EXPECTED_PANELS)
    def test_panel_is_a_single_self_contained_svg(self, fig2_run, panel_name):
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
@pytest.mark.skipif(
    not _HAS_ALL_CWC_CANDIDATE_DATA,
    reason=(
        "real model-states artifacts not found under "
        f"{MODEL_STATES_DIR} for one or more of {sorted(CWC_DATASET_CHOICES)}"
    ),
)
class TestDeterminism:
    def test_second_run_is_byte_identical(self, fig2_run, tmp_path_factory):
        result, _cache_dir, env, _started = fig2_run
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
        # CWC sample get served from disk on the second run, making the
        # byte-identity check pass without the seed actually doing anything.
        second_cache_dir = tmp_path_factory.mktemp("fig_transforms_cache_second")
        second_env = dict(env)
        second_env["LIC_FIG_CACHE_DIR"] = str(second_cache_dir)

        second = subprocess.run(
            [sys.executable, str(FIG2_SCRIPT)],
            cwd=REPO_ROOT,
            env=second_env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert second.returncode == 0, (
            "second headless run of figures/fig2_ideal_observer.py exited "
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
            "-- check that every source of randomness (the CWC transforms' "
            "sampling in particular) is seeded by an explicit constant"
        )

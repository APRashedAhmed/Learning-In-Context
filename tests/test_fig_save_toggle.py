"""Static contract tests for the figure scripts' save/display toggle.

Each of the four paper-figure marimo notebooks under ``figures/`` — fig4, fig5,
fig6, fig7 — is expected to expose a ``save_svgs`` marimo UI switch and to use
it to gate every SVG export, while still displaying the rendered figure(s)
inline so `marimo edit` remains useful as a review surface. Concretely, each
script must satisfy three properties:

1. It defines ``save_svgs = mo.ui.switch(value=True, ...)`` somewhere in the
   module — defaulting to ``True`` so a headless ``python figures/fig*.py``
   run (which calls ``app.run()`` and never touches the UI) still saves every
   panel, matching today's behavior.
2. Every ``paper_style.save_panel(...)`` call is lexically nested inside an
   ``if save_svgs.value:`` block, so flipping the switch off in an interactive
   session skips the disk write.
3. Every marimo cell function (a top-level ``def _(...):`` decorated with
   ``@app.cell``) that calls ``save_panel`` ends, as its last statement before
   the cell's trailing ``return``, with a bare expression statement that
   actually displays something — the figure object itself, or an
   ``mo.vstack([...])`` of figures for cells that render several panels in a
   loop. A call that merely re-invokes a private throwaway closure (the
   ``def _(): ...; _()`` idiom used today, which discards its return value and
   therefore displays nothing) does not satisfy this — nor does ending on the
   ``save_panel`` call itself.

These are pure ``ast`` checks over the script source: no marimo runtime, no
figure rendering, no data access. They are written before the feature lands
and are expected to fail against the current scripts, which have no
``save_svgs`` switch, no gating, and end every save-bearing cell on a
throwaway closure call rather than a display expression.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"

FIGURE_SCRIPTS = [
    FIGURES_DIR / "fig4_identifying_units.py",
    FIGURES_DIR / "fig5_unit_activity.py",
    FIGURES_DIR / "fig6_interventions.py",
    FIGURES_DIR / "fig7_gates.py",
]


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _is_save_panel_call(node: ast.AST) -> bool:
    """True if `node` is a Call to `<something>.save_panel(...)`."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "save_panel"
    )


def _find_save_panel_calls(tree: ast.Module) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if _is_save_panel_call(node)]


def _find_save_svgs_switch(tree: ast.Module) -> ast.Assign | None:
    """Find `save_svgs = mo.ui.switch(value=True, ...)` anywhere in the module."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t for t in node.targets if isinstance(t, ast.Name)]
        if not any(t.id == "save_svgs" for t in targets):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        if not (isinstance(call.func, ast.Attribute) and call.func.attr == "switch"):
            continue
        # call.func should resolve to `mo.ui.switch` (an Attribute chain
        # ending in `.ui.switch` off a `mo` name).
        inner = call.func.value
        if not (isinstance(inner, ast.Attribute) and inner.attr == "ui"):
            continue
        has_true_value_kw = any(
            kw.arg == "value" and isinstance(kw.value, ast.Constant) and kw.value.value is True
            for kw in call.keywords
        )
        if has_true_value_kw:
            return node
    return None


def _has_matching_if_ancestor(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Walk up from `node` looking for an enclosing `if save_svgs.value:` block."""
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.If) and _is_save_svgs_value_test(current.test):
            return True
        current = parents.get(current)
    return False


def _is_save_svgs_value_test(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Attribute)
        and test.attr == "value"
        and isinstance(test.value, ast.Name)
        and test.value.id == "save_svgs"
    )


def _is_marimo_cell_function(node: ast.AST) -> bool:
    """Top-level `def _(...):` carrying an `@app.cell` (or `@app.cell(...)`) decorator."""
    if not isinstance(node, ast.FunctionDef) or node.name != "_":
        return False
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and target.attr == "cell":
            return True
    return False


def _find_marimo_cell_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    return [node for node in tree.body if _is_marimo_cell_function(node)]


def _last_non_return_statement(func: ast.FunctionDef) -> ast.stmt | None:
    body = [stmt for stmt in func.body if not isinstance(stmt, ast.Return)]
    return body[-1] if body else None


def _is_display_expression(stmt: ast.stmt) -> bool:
    """A bare `ast.Expr` that plausibly displays a figure or `mo.vstack([...])`.

    Explicitly rejects the `save_panel(...)` call itself and the throwaway
    `_()` closure-invocation idiom used by the pre-feature scripts, since
    neither actually surfaces a figure to marimo's cell output.
    """
    if not isinstance(stmt, ast.Expr):
        return False
    value = stmt.value
    if _is_save_panel_call(value):
        return False
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "_":
        return False
    return True


@pytest.mark.parametrize("script_path", FIGURE_SCRIPTS, ids=lambda p: p.stem)
class TestSaveSvgsToggleContract:
    """Contract 1: a `save_svgs = mo.ui.switch(value=True, ...)` switch exists."""

    def test_save_svgs_switch_defined_default_true(self, script_path: Path):
        tree = _parse(script_path)
        switch = _find_save_svgs_switch(tree)
        assert switch is not None, (
            f"{script_path.name}: expected a `save_svgs = mo.ui.switch(value=True, ...)` "
            "assignment somewhere in the module; none found."
        )


@pytest.mark.parametrize("script_path", FIGURE_SCRIPTS, ids=lambda p: p.stem)
class TestSavePanelGatingContract:
    """Contract 2: every `save_panel(...)` call is inside `if save_svgs.value:`."""

    def test_every_save_panel_call_is_gated(self, script_path: Path):
        tree = _parse(script_path)
        parents = _build_parent_map(tree)
        calls = _find_save_panel_calls(tree)
        assert calls, f"{script_path.name}: expected at least one save_panel call to check."

        ungated = [
            call for call in calls if not _has_matching_if_ancestor(call, parents)
        ]
        assert not ungated, (
            f"{script_path.name}: {len(ungated)} save_panel call(s) (of {len(calls)}) "
            f"at line(s) {[c.lineno for c in ungated]} are not lexically inside an "
            "`if save_svgs.value:` block."
        )


@pytest.mark.parametrize("script_path", FIGURE_SCRIPTS, ids=lambda p: p.stem)
class TestInlineDisplayContract:
    """Contract 3: save-bearing cells end on a real display expression."""

    def test_save_bearing_cells_end_with_display_expression(self, script_path: Path):
        tree = _parse(script_path)
        cell_functions = _find_marimo_cell_functions(tree)
        assert cell_functions, f"{script_path.name}: expected marimo cell functions to check."

        save_bearing = [
            fn for fn in cell_functions if _find_save_panel_calls(fn)
        ]
        assert save_bearing, (
            f"{script_path.name}: expected at least one cell containing a save_panel call."
        )

        offenders = []
        for fn in save_bearing:
            last_stmt = _last_non_return_statement(fn)
            if last_stmt is None or not _is_display_expression(last_stmt):
                offenders.append(fn.lineno)

        assert not offenders, (
            f"{script_path.name}: cell(s) starting at line(s) {offenders} contain a "
            "save_panel call but do not end with a display expression (a bare figure "
            "or `mo.vstack([...])`) before their trailing `return` — they end on the "
            "save_panel call itself or on a throwaway closure invocation instead."
        )

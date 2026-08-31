"""Static contract tests for the figure scripts' save/display toggle.

Each of the six paper-figure marimo notebooks under ``figures/`` — fig2, fig3,
fig4, fig5, fig6, fig7 — is expected to expose a ``save_svgs`` marimo UI switch and to use
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
   loop. A call that merely re-invokes a private throwaway closure (the bare
   ``def _(): ...; _()`` idiom, which discards its return value and therefore
   displays nothing) does not satisfy this — nor does ending on the
   ``save_panel`` call itself, nor on an inert literal.
4. The switch is wired up the way marimo's file format demands: the cell that
   assigns ``save_svgs`` also returns it, and every *other* cell that reads
   ``save_svgs`` declares it as a parameter. Without both halves the notebook
   raises ``NameError`` at cell-run time even though contracts 1-3 hold.

These are pure ``ast`` checks over the script source: no marimo runtime, no
figure rendering, no data access.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"

FIGURE_SCRIPTS = [
    FIGURES_DIR / "fig2_ideal_observer.py",
    FIGURES_DIR / "fig3_task_results.py",
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
        # call.func must resolve to `mo.ui.switch` exactly — an Attribute chain
        # `switch` <- `ui` <- the `mo` Name, so an unrelated `foo.ui.switch(...)`
        # does not satisfy the contract.
        inner = call.func.value
        if not (isinstance(inner, ast.Attribute) and inner.attr == "ui"):
            continue
        if not (isinstance(inner.value, ast.Name) and inner.value.id == "mo"):
            continue
        has_true_value_kw = any(
            kw.arg == "value" and isinstance(kw.value, ast.Constant) and kw.value.value is True
            for kw in call.keywords
        )
        if has_true_value_kw:
            return node
    return None


def _has_matching_if_ancestor(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Walk up from `node` looking for an enclosing `if save_svgs.value:` block.

    Only the `if`'s *body* counts: a call reached through the `orelse` branch
    runs precisely when the switch is off, so `else:`/`elif` placement (marked
    by `orelse` in the AST) is not gating and must not pass.
    """
    child, current = node, parents.get(node)
    while current is not None:
        if (
            isinstance(current, ast.If)
            and _is_save_svgs_value_test(current.test)
            and any(stmt is child for stmt in current.body)
        ):
            return True
        child, current = current, parents.get(current)
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
    """A bare `ast.Expr` that actually surfaces a figure as the cell's output.

    Accepted: a reference to an already-built figure — a name (`_fig`), an
    attribute or subscript (`_figs[0]`), or a tuple/list of those — or a call on
    `mo` such as `mo.vstack([...])` for cells rendering several panels.

    Rejected: the `save_panel(...)` call itself (returns a `Path`), the
    throwaway `def _(): ...; _()` closure-invocation idiom (discards its return
    value), and inert literals such as a bare string or `None`, none of which
    display a figure.
    """
    if not isinstance(stmt, ast.Expr):
        return False
    value = stmt.value
    if isinstance(value, (ast.Name, ast.Attribute, ast.Subscript, ast.Tuple, ast.List)):
        return True
    # `mo.vstack([...])` / `mo.hstack([...])` and friends: a call whose receiver
    # chain bottoms out at the `mo` name.
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
        if _is_save_panel_call(value):
            return False
        root = value.func
        while isinstance(root, ast.Attribute):
            root = root.value
        return isinstance(root, ast.Name) and root.id == "mo"
    return False


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


def _cell_parameter_names(func: ast.FunctionDef) -> set[str]:
    args = func.args
    return {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}


def _assigns_name(func: ast.FunctionDef, name: str) -> bool:
    return any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        for node in ast.walk(func)
    )


def _returned_names(func: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for stmt in func.body:
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            names |= {
                node.id for node in ast.walk(stmt.value) if isinstance(node, ast.Name)
            }
    return names


def _reads_name(func: ast.FunctionDef, name: str) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
        for node in ast.walk(func)
    )


@pytest.mark.parametrize("script_path", FIGURE_SCRIPTS, ids=lambda p: p.stem)
class TestSwitchCellWiringContract:
    """Contract 4: `save_svgs` is wired per marimo's file format.

    marimo cells are plain functions: a cell's globals arrive as parameters and
    its definitions leave via the trailing `return`. A switch that is defined
    but not returned, or read by a cell that does not declare it, raises
    `NameError` when that cell runs — a failure the other three contracts, which
    only look at lexical structure, cannot see.
    """

    def test_defining_cell_returns_the_switch(self, script_path: Path):
        tree = _parse(script_path)
        definers = [
            fn for fn in _find_marimo_cell_functions(tree) if _assigns_name(fn, "save_svgs")
        ]
        assert definers, (
            f"{script_path.name}: no marimo cell assigns `save_svgs`; the switch must "
            "live in a cell so marimo can expose it to the render cells."
        )
        offenders = [fn.lineno for fn in definers if "save_svgs" not in _returned_names(fn)]
        assert not offenders, (
            f"{script_path.name}: the cell(s) at line(s) {offenders} assign `save_svgs` "
            "but do not return it, so no other cell can read the switch."
        )

    def test_consuming_cells_declare_the_switch_parameter(self, script_path: Path):
        tree = _parse(script_path)
        offenders = [
            fn.lineno
            for fn in _find_marimo_cell_functions(tree)
            if _reads_name(fn, "save_svgs")
            and not _assigns_name(fn, "save_svgs")
            and "save_svgs" not in _cell_parameter_names(fn)
        ]
        assert not offenders, (
            f"{script_path.name}: the cell(s) at line(s) {offenders} read `save_svgs` "
            "without declaring it as a cell parameter — they would raise NameError."
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

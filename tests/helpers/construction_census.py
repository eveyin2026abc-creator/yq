# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""List test call sites that still construct a model, or already use the executor.

The scan is an AST walk of ``tests/``. A name that only appears inside a string
is not a call, so the layer-guard fixtures do not show up as constructions.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_TESTS = _REPO / "tests"

CONSTRUCTION_CALLEES = frozenset(
    {
        "ModelRunner",
        "build_model",
        "get_cached_build_model",
        "TransformerModel",
        "isolated_l1_executor",
        "L1ScenarioExecutor",
        "capture_artifact_for_profile",
    }
)

_DIRECT_MODEL = frozenset({"ModelRunner", "build_model", "get_cached_build_model", "TransformerModel"})
_EXECUTOR = frozenset({"isolated_l1_executor", "L1ScenarioExecutor"})


@dataclass(frozen=True)
class ConstructionSite:
    path: str
    function: str
    line: int
    callee: str

    @property
    def direct_model(self) -> bool:
        return self.callee in _DIRECT_MODEL

    @property
    def executor(self) -> bool:
        return self.callee in _EXECUTOR


def _callee_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _function_stack(tree: ast.AST) -> dict[ast.AST, str]:
    parents: dict[ast.AST, str] = {}

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._stack: list[str] = ["<module>"]

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._stack.append(node.name)
            self.generic_visit(node)
            self._stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._stack.append(node.name)
            self.generic_visit(node)
            self._stack.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._stack.append(node.name)
            self.generic_visit(node)
            self._stack.pop()

        def generic_visit(self, node: ast.AST) -> None:
            parents[node] = self._stack[-1]
            super().generic_visit(node)

    _Visitor().visit(tree)
    return parents


def iter_sites(root: Path | None = None) -> tuple[ConstructionSite, ...]:
    """Return every tracked construction call under ``tests/``, path-sorted."""

    base = _TESTS if root is None else root
    sites: list[ConstructionSite] = []
    for path in sorted(base.rglob("*.py")):
        if "assets" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        parents = _function_stack(tree)
        relative = path.relative_to(_REPO).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _callee_name(node)
            if name not in CONSTRUCTION_CALLEES:
                continue
            sites.append(
                ConstructionSite(
                    path=relative,
                    function=parents.get(node, "<module>"),
                    line=node.lineno,
                    callee=name,
                )
            )
    return tuple(sorted(sites, key=lambda site: (site.path, site.line, site.callee)))


def sites_in(relative_path: str, sites: tuple[ConstructionSite, ...] | None = None) -> tuple[ConstructionSite, ...]:
    found = iter_sites() if sites is None else sites
    return tuple(site for site in found if site.path == relative_path)


def render_census(sites: tuple[ConstructionSite, ...] | None = None) -> str:
    """A line-oriented register of the call sites. Stable enough to diff."""

    found = iter_sites() if sites is None else sites
    lines = [
        "# L1 construction census",
        "",
        "Call sites under tests/ that construct a model or already go through the L1 executor.",
        "String mentions are not included. Node ids are unchanged.",
        "",
        "| path | function | line | callee |",
        "| --- | --- | --- | --- |",
    ]
    for site in found:
        lines.append(f"| {site.path} | {site.function} | {site.line} | {site.callee} |")
    lines.append("")
    direct = sum(1 for site in found if site.direct_model)
    executor = sum(1 for site in found if site.executor)
    capture = sum(1 for site in found if site.callee == "capture_artifact_for_profile")
    lines.append(f"Direct model constructions: {direct}.")
    lines.append(f"Executor constructions: {executor}.")
    lines.append(f"Diagnostics captures: {capture}.")
    lines.append("")
    return "\n".join(lines)

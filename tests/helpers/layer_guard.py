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

"""AST check that an L2 module does not build a real model."""

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN_NAMES = frozenset({"ModelRunner", "build_model", "TransformerModel"})


def _called_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def find_l2_model_violations(source: str, *, filename: str = "<l2>") -> tuple[str, ...]:
    """Return one message per forbidden model-construction use in *source*."""
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []
    for function in (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
        calls_forbidden = any(
            isinstance(child, ast.Call) and _called_name(child.func) in _FORBIDDEN_NAMES
            for child in ast.walk(function)
        )
        if calls_forbidden:
            for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs):
                if argument.arg == "model_id":
                    violations.append(f"{filename}:{argument.lineno}: parameter model_id")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = _called_name(node.func)
        if called in _FORBIDDEN_NAMES:
            violations.append(f"{filename}:{node.lineno}: uses {called}")
        for keyword in node.keywords:
            if keyword.arg == "model_id":
                violations.append(f"{filename}:{node.lineno}: keyword model_id")
    return tuple(violations)


def find_l2_model_violations_in_file(path: Path) -> tuple[str, ...]:
    """Scan one Python file."""
    return find_l2_model_violations(path.read_text(encoding="utf-8"), filename=str(path))

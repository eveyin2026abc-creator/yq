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

"""AST check that an L2 module does not build a real model.

This scanner is not wired into ``run_ci_gate.sh``. Callers run it on the L2
helper module. It stays off the product path until the migration is approved.
"""

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN_NAMES = frozenset({"ModelRunner", "build_model", "TransformerModel"})


def find_l2_model_violations(source: str, *, filename: str = "<l2>") -> tuple[str, ...]:
    """Return one message per forbidden model-construction use in *source*."""
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            violations.append(f"{filename}:{node.lineno}: uses {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_NAMES:
            violations.append(f"{filename}:{node.lineno}: uses {node.attr}")
        elif isinstance(node, ast.arg) and node.arg == "model_id":
            violations.append(f"{filename}:{node.lineno}: parameter model_id")
        elif isinstance(node, ast.keyword) and node.arg == "model_id":
            violations.append(f"{filename}:{node.lineno}: keyword model_id")
    return tuple(violations)


def find_l2_model_violations_in_file(path: Path) -> tuple[str, ...]:
    """Scan one Python file."""
    return find_l2_model_violations(path.read_text(encoding="utf-8"), filename=str(path))

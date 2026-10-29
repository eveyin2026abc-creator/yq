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
"""Split nightly line coverage into core and cov_helper contributions."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CoverageBreakdown:
    core_line_percent: float
    cov_helper_line_contribution: float
    cov_helper_only_lines: int


def collect_cov_helper_nodes(repo_root: Path, python_exe: str) -> set[str]:
    """Return exact node IDs carrying the cov_helper marker."""
    command = [
        python_exe,
        "-m",
        "pytest",
        "tests",
        "--collect-only",
        "-q",
        "-m",
        "cov_helper",
        "-o",
        "addopts=",
    ]
    completed = subprocess.run(command, cwd=repo_root, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode not in {0, 5}:
        raise RuntimeError(f"cov_helper collection failed: {completed.stderr.strip()}")
    return {line.strip() for line in completed.stdout.splitlines() if line.startswith("tests/") and "::" in line}


def _node_from_context(context: str) -> str:
    return context.split("|", 1)[0]


def split_line_coverage(data_file: Path, helper_nodes: set[str]) -> CoverageBreakdown:
    """Compute core percent and helper-only increment from coverage contexts."""
    from coverage import Coverage

    cov = Coverage(data_file=str(data_file))
    try:
        cov.load()
    except Exception as error:
        raise RuntimeError(f"cannot read coverage data: {error}") from error
    data = cov.get_data()
    total_statements = 0
    all_covered: set[tuple[str, int]] = set()
    core_covered: set[tuple[str, int]] = set()
    for filename in data.measured_files():
        try:
            statements = cov.analysis2(filename)[1]
        except Exception:
            continue
        total_statements += len(statements)
        contexts = data.contexts_by_lineno(filename)
        for line, line_contexts in contexts.items():
            key = (filename, line)
            all_covered.add(key)
            if not line_contexts or any(_node_from_context(context) not in helper_nodes for context in line_contexts):
                core_covered.add(key)
    if total_statements <= 0:
        raise RuntimeError("coverage data has no measurable statements")
    helper_only = all_covered - core_covered
    return CoverageBreakdown(
        core_line_percent=100.0 * len(core_covered) / total_statements,
        cov_helper_line_contribution=100.0 * len(helper_only) / total_statements,
        cov_helper_only_lines=len(helper_only),
    )


def build_coverage_breakdown(repo_root: Path, python_exe: str) -> CoverageBreakdown:
    return split_line_coverage(repo_root / ".coverage", collect_cov_helper_nodes(repo_root, python_exe))

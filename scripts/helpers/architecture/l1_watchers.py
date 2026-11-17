#!/usr/bin/env python3

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

"""Run the incremental L1 families and write their watcher evidence artifact."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.common.build_test_map import collect_from_coverage, write_test_map
from scripts.helpers.common.coverage_config import cov_pytest_args
from scripts.helpers.common.pytest_runner import PYTEST_IGNORE_ADDOPTS

_FAMILIES = frozenset({"qwen", "dense", "moe", "multimodal"})


def required_nodes(watchers_path: Path) -> tuple[str, ...]:
    """Return one stable, required L1 source node for every registered family."""
    with watchers_path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(csv.DictReader(stream))
    nodes_by_family: dict[str, str] = {}
    for row in rows:
        if row.get("required", "").lower() != "true":
            continue
        family = row.get("family", "")
        node = row.get("source_node", "")
        if family not in _FAMILIES or not node:
            continue
        previous = nodes_by_family.setdefault(family, node)
        if previous != node:
            raise ValueError(f"family {family!r} has multiple required source nodes")
    missing = sorted(_FAMILIES - nodes_by_family.keys())
    if missing:
        raise ValueError(f"watcher manifest is missing required families: {', '.join(missing)}")
    return tuple(nodes_by_family[family] for family in sorted(_FAMILIES))


def pytest_command(nodes: tuple[str, ...]) -> list[str]:
    """Build the deterministic, non-nightly L1 evidence command."""
    return [
        sys.executable,
        "-m",
        "pytest",
        *PYTEST_IGNORE_ADDOPTS,
        *nodes,
        "-m",
        "not nightly",
        "-q",
        "--tb=short",
        *cov_pytest_args(cov_context=True),
    ]


def generate_candidate(watchers_path: Path, output_path: Path, coverage_path: Path) -> int:
    """Execute the four incremental families and persist their observed mapping."""
    nodes = required_nodes(watchers_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.unlink(missing_ok=True)

    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(coverage_path.resolve())
    result = subprocess.run(pytest_command(nodes), cwd=REPO_ROOT, env=env, check=False)
    if result.returncode != 0:
        return result.returncode

    mapping = collect_from_coverage(frozenset(nodes), coverage_path=coverage_path)
    write_test_map(output_path, mapping)
    print(f"L1 watcher evidence: {output_path} ({len(mapping)} family nodes)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchers", type=Path, default=Path("tests/.ci/l1_watchers.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("MSMODELING_L1_CANDIDATE_MAP_PATH", "artifacts/l1_test_map.json")),
    )
    parser.add_argument("--coverage-file", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    coverage_path = args.coverage_file or args.output.with_name(".coverage.l1")
    try:
        return generate_candidate(args.watchers, args.output, coverage_path)
    except (OSError, ValueError) as error:
        print(f"L1 watcher evidence failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

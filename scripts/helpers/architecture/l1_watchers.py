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
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.common.build_test_map import collect_from_coverage, write_test_map
from scripts.helpers.common.coverage_config import cov_pytest_args
from scripts.helpers.common.pytest_runner import PYTEST_IGNORE_ADDOPTS
from scripts.helpers.architecture.l1_contracts import contract_rows

_FAMILIES = frozenset({"qwen", "dense", "moe", "multimodal"})
_ALL_FAMILY_PATHS = frozenset(
    {
        "scripts/helpers/architecture/l1_watchers.py",
        "scripts/helpers/architecture/main.py",
        "scripts/run_ci_gate.sh",
        "tests/.ci/l1_contracts.json",
        "tests/helpers/l1_scenario.py",
        "tests/helpers/model_builder.py",
        "tests/helpers/model_cache.py",
        "tests/regression/tensor_cast/model/test_l1_family_scenarios.py",
    }
)
_QWEN_PATHS = frozenset(
    {
        "tests/regression/tensor_cast/model/test_l1_qwen.py",
        "tests/scenarios/l1/l1_qwen.py",
    }
)


def _watcher_rows(path: Path) -> tuple[dict[str, str], ...]:
    if path.suffix == ".json":
        return tuple(contract_rows(path, "watchers"))
    with path.open(newline="", encoding="utf-8") as stream:
        return tuple(csv.DictReader(stream))


def required_nodes(watchers_path: Path, families: frozenset[str] = _FAMILIES) -> tuple[str, ...]:
    """Return one stable, required L1 source node for every registered family."""
    rows = _watcher_rows(watchers_path)
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
    missing = sorted(families - nodes_by_family.keys())
    if missing:
        raise ValueError(f"watcher manifest is missing required families: {', '.join(missing)}")
    return tuple(nodes_by_family[family] for family in sorted(families))


def select_families(watchers_path: Path, changed_paths: tuple[str, ...]) -> tuple[frozenset[str], dict[str, list[str]]]:
    """Select only L1 families affected by this PR's paths."""
    rows = _watcher_rows(watchers_path)
    product_families: dict[str, set[str]] = {}
    for row in rows:
        if row.get("required", "").lower() != "true":
            continue
        product_path = row.get("product_symbol", "").split("::", 1)[0]
        product_families.setdefault(product_path, set()).add(row.get("family", ""))

    selected: set[str] = set()
    reasons: dict[str, list[str]] = {family: [] for family in sorted(_FAMILIES)}
    for raw_path in changed_paths:
        path = raw_path.strip().replace("\\", "/")
        if not path:
            continue
        if path in _ALL_FAMILY_PATHS:
            for family in _FAMILIES:
                selected.add(family)
                reasons[family].append(path)
            continue
        if path in _QWEN_PATHS:
            selected.add("qwen")
            reasons["qwen"].append(path)
        for family in product_families.get(path, set()):
            if family in _FAMILIES:
                selected.add(family)
                reasons[family].append(path)
        candidate = REPO_ROOT / path
        if path.endswith(".py") and candidate.is_file() and path.startswith("tests/"):
            source = candidate.read_text(encoding="utf-8")
            for family in re.findall(r"L1_FAMILY:\s*([a-z0-9_-]+)", source):
                if family in _FAMILIES:
                    selected.add(family)
                    reasons[family].append(path)
    return frozenset(selected), {family: sorted(set(reasons[family])) for family in sorted(selected)}


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


def generate_candidate(
    watchers_path: Path,
    output_path: Path,
    coverage_path: Path,
    families: frozenset[str] = _FAMILIES,
) -> int:
    """Execute affected incremental families and persist their observed mapping."""
    nodes = required_nodes(watchers_path, families)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.unlink(missing_ok=True)

    if not nodes:
        write_test_map(output_path, {})
        print(f"L1 watcher evidence skipped: no affected family ({output_path})")
        return 0

    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(coverage_path.resolve())
    result = subprocess.run(pytest_command(nodes), cwd=REPO_ROOT, env=env, check=False)
    if result.returncode != 0:
        return result.returncode

    mapping = collect_from_coverage(frozenset(nodes), coverage_path=coverage_path)
    write_test_map(output_path, mapping)
    print(f"L1 watcher evidence: {output_path} ({len(mapping)} affected family nodes)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchers", type=Path, default=Path("tests/.ci/l1_contracts.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("MSMODELING_L1_CANDIDATE_MAP_PATH", "artifacts/l1_test_map.json")),
    )
    parser.add_argument("--coverage-file", type=Path)
    parser.add_argument("--changed-paths-file", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, default=Path("artifacts/l1_watcher_selection.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    coverage_path = args.coverage_file or args.output.with_name(".coverage.l1")
    try:
        changed_paths = tuple(args.changed_paths_file.read_text(encoding="utf-8").splitlines())
        families, reasons = select_families(args.watchers, changed_paths)
        args.selection_output.parent.mkdir(parents=True, exist_ok=True)
        args.selection_output.write_text(
            json.dumps(
                {"schema_version": 1, "families": sorted(families), "reasons": reasons},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return generate_candidate(args.watchers, args.output, coverage_path, families)
    except (OSError, ValueError) as error:
        print(f"L1 watcher evidence failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

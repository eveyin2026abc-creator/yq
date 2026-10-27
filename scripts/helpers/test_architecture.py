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
"""Mechanical gates for the L1/L2/L3 test architecture."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ASSERTION_GROUPS = {"numeric", "structural", "capability"}
INVENTORY_FIELDS = ("node_id", "current_file", "current_layer", "target_layer", "target_file", "status")


class GateError(ValueError):
    """A reviewable architecture contract was violated."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _require_columns(path: Path, rows: list[dict[str, str]], required: Iterable[str]) -> None:
    if rows:
        columns = set(rows[0])
    else:
        with path.open(newline="", encoding="utf-8") as stream:
            columns = set(next(csv.reader(stream), []))
    missing = set(required) - columns
    if missing:
        raise GateError(f"{path}: missing columns: {', '.join(sorted(missing))}")


def _write_csv(path: Path, rows: Iterable[dict[str, str]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _node_ids(path: Path) -> list[str]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value.startswith("tests/") and "::" in value:
            # Parameter IDs may legitimately contain spaces (for example YAML
            # snippets).  The complete pytest output line is the node ID.
            result.append(value)
    return sorted(set(result))


def collect_inventory(args: argparse.Namespace) -> None:
    rows = []
    for node_id in _node_ids(args.collect_output):
        current_file = node_id.split("::", 1)[0]
        rows.append({
            "node_id": node_id,
            "current_file": current_file,
            "current_layer": "unclassified",
            "target_layer": "pending-review",
            "target_file": current_file,
            "status": "master-baseline",
        })
    _write_csv(args.output, rows, INVENTORY_FIELDS)
    print(json.dumps({"collected": len(rows), "output": str(args.output)}))


def compare_inventory(args: argparse.Namespace) -> None:
    rows = _read_csv(args.inventory)
    _require_columns(args.inventory, rows, INVENTORY_FIELDS)
    inventory = {row["node_id"] for row in rows}
    collected = set(_node_ids(args.collect_output))
    missing = sorted(collected - inventory)
    extra = sorted(inventory - collected)
    renamed: list[dict[str, str]] = []
    by_tail: dict[str, list[str]] = defaultdict(list)
    for node in extra:
        by_tail[node.split("::", 1)[-1]].append(node)
    for node in missing:
        matches = by_tail.get(node.split("::", 1)[-1], [])
        if len(matches) == 1:
            renamed.append({"old_node_id": matches[0], "new_node_id": node})
    _write_csv(args.output_dir / "inventory_missing.csv", ({"node_id": item} for item in missing), ("node_id",))
    _write_csv(args.output_dir / "inventory_extra.csv", ({"node_id": item} for item in extra), ("node_id",))
    _write_csv(args.output_dir / "inventory_renamed.csv", renamed, ("old_node_id", "new_node_id"))
    print(json.dumps({"missing": len(missing), "extra": len(extra), "renamed": len(renamed)}))
    if missing or extra:
        raise GateError("test inventory drifted; inspect the three CSV diff reports")


def check_assertions(args: argparse.Namespace) -> None:
    required = ("family", "group", "assertion_id", "source_node", "product_symbol", "status", "reason", "approver")
    rows = _read_csv(args.manifest)
    _require_columns(args.manifest, rows, required)
    seen: set[tuple[str, str]] = set()
    groups: dict[str, set[str]] = defaultdict(set)
    active: set[tuple[str, str]] = set()
    for line, row in enumerate(rows, 2):
        key = (row["family"], row["assertion_id"])
        if not all(key) or key in seen:
            raise GateError(f"{args.manifest}:{line}: empty or duplicate family/assertion_id")
        seen.add(key)
        if row["group"] not in ASSERTION_GROUPS:
            raise GateError(f"{args.manifest}:{line}: invalid assertion group {row['group']!r}")
        if not row["source_node"].startswith("tests/") or "::" not in row["source_node"]:
            raise GateError(f"{args.manifest}:{line}: source_node must be a concrete pytest node")
        if row["status"] not in {"active", "waived"}:
            raise GateError(f"{args.manifest}:{line}: status must be active or waived")
        if row["status"] == "waived" and not (row["reason"] and row["approver"]):
            raise GateError(f"{args.manifest}:{line}: waived assertion requires reason and approver")
        groups[row["family"]].add(row["group"])
        if row["status"] == "active":
            active.add(key)
    incomplete = {family: sorted(ASSERTION_GROUPS - values) for family, values in groups.items() if values != ASSERTION_GROUPS}
    if incomplete:
        raise GateError(f"every family needs numeric/structural/capability assertions: {incomplete}")
    if args.baseline:
        baseline = _read_csv(args.baseline)
        _require_columns(args.baseline, baseline, required)
        baseline_active = {(r["family"], r["assertion_id"]) for r in baseline if r["status"] == "active"}
        waived = {(r["family"], r["assertion_id"]) for r in rows if r["status"] == "waived" and r["reason"] and r["approver"]}
        lost = sorted(baseline_active - active - waived)
        if lost:
            raise GateError(f"assertions disappeared without signed waiver: {lost}")
    if args.inventory and args.inventory.exists():
        inventory_rows = _read_csv(args.inventory)
        _require_columns(args.inventory, inventory_rows, INVENTORY_FIELDS)
        inventory_nodes = {row["node_id"] for row in inventory_rows}
        unknown = sorted(row["source_node"] for row in rows if row["status"] == "active" and row["source_node"] not in inventory_nodes)
        if unknown:
            raise GateError(f"active assertions reference nodes outside the master inventory: {unknown}")
    print(json.dumps({"families": len(groups), "active_assertions": len(active)}))


def check_scenarios(args: argparse.Namespace) -> None:
    required = ("family", "scenario_id", "build_signature", "run_signature", "expected_seconds", "schedule")
    rows = _read_csv(args.matrix)
    _require_columns(args.matrix, rows, required)
    invalid = sorted({row["schedule"] for row in rows} - {"incremental", "nightly", "pending-approval"})
    if invalid:
        raise GateError(f"invalid schedules: {invalid}")
    scenarios: dict[tuple[str, str], float] = {}
    builds = set()
    for line, row in enumerate((r for r in rows if r["schedule"] == "incremental"), 2):
        try:
            seconds = float(row["expected_seconds"])
        except ValueError as error:
            raise GateError(f"{args.matrix}:{line}: expected_seconds is not numeric") from error
        key = (row["build_signature"], row["run_signature"])
        scenarios[key] = max(scenarios.get(key, 0.0), seconds)
        builds.add(row["build_signature"])
    total = sum(scenarios.values())
    print(json.dumps({"builds": len(builds), "runs": len(scenarios), "expected_seconds": round(total, 3), "budget_seconds": args.budget}))
    if total > args.budget:
        raise GateError(f"incremental L1 scenario budget exceeded: {total:.3f}s > {args.budget:.3f}s")


def _changed_test_files(base: str) -> list[Path]:
    # Existing flat tests are migrated family-by-family.  Enforce the new
    # contract on added files without turning an unrelated edit into a forced
    # whole-file migration.
    command = ["git", "diff", "--name-only", "--diff-filter=A", base, "--", "tests"]
    output = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE).stdout
    return [Path(line) for line in output.splitlines() if line.endswith(".py") and Path(line).exists()]


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _name(node.func)
    return ""


def check_new_tests(args: argparse.Namespace) -> None:
    registry = json.loads(args.family_registry.read_text(encoding="utf-8"))
    registered = set(registry.get("families", {}))
    errors = []
    files = _changed_test_files(args.base)
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        names = {_name(node) for node in ast.walk(tree) if isinstance(node, (ast.Name, ast.Attribute))}
        uses_model_execution = "ModelRunner" in names or "build_model" in names
        uses_executor = "L1ScenarioExecutor" in names or "l1_executor" in source
        if uses_model_execution and not uses_executor and "tests/helpers/tests/" not in path.as_posix():
            errors.append(f"{path}: L1 model execution must use L1ScenarioExecutor/l1_executor")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
                continue
            decorators = {_name(item) for item in node.decorator_list}
            if any(name.endswith("cov_helper") for name in decorators):
                segment = ast.get_source_segment(source, node) or ""
                if "COV_HELPER:" not in (ast.get_docstring(node) or ""):
                    errors.append(f"{path}:{node.lineno}: cov_helper needs a COV_HELPER: product/path.py::symbol docstring")
                if any(token in segment for token in ('patch("tensor_cast.', 'patch("cli.', 'patch("serving_cast.')):
                    errors.append(f"{path}:{node.lineno}: cov_helper must not mock the product object under test")
        if uses_model_execution and registered and not any(f"L1_FAMILY: {family}" in source for family in registered):
            errors.append(f"{path}: L1 test must declare a registered 'L1_FAMILY: <name>'")
    if errors:
        raise GateError("\n".join(errors))
    print(json.dumps({"checked_files": len(files)}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect-inventory")
    collect.add_argument("--collect-output", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.set_defaults(func=collect_inventory)
    compare = sub.add_parser("compare-inventory")
    compare.add_argument("--inventory", type=Path, required=True)
    compare.add_argument("--collect-output", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.set_defaults(func=compare_inventory)
    assertions = sub.add_parser("check-assertions")
    assertions.add_argument("--manifest", type=Path, default=Path("tests/.ci/assertion_manifest.csv"))
    assertions.add_argument("--baseline", type=Path)
    assertions.add_argument("--inventory", type=Path, default=Path("tests/.ci/inventory/master_test_inventory.csv"))
    assertions.set_defaults(func=check_assertions)
    scenarios = sub.add_parser("check-scenarios")
    scenarios.add_argument("--matrix", type=Path, default=Path("tests/.ci/l1_scenarios.csv"))
    scenarios.add_argument("--budget", type=float, default=180.0)
    scenarios.set_defaults(func=check_scenarios)
    new_tests = sub.add_parser("check-new-tests")
    new_tests.add_argument("--base", default="origin/master")
    new_tests.add_argument("--family-registry", type=Path, default=Path("tests/.ci/l1_families.json"))
    new_tests.set_defaults(func=check_new_tests)
    return parser


def main() -> int:
    try:
        args = build_parser().parse_args()
        args.func(args)
    except (GateError, OSError, subprocess.CalledProcessError) as error:
        print(f"test architecture gate: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

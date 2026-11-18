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

from tests.architecture.layer_guard import find_l2_model_violations_in_file

ASSERTION_GROUPS = {"numeric", "structural", "capability"}
L2_BUILDERS = (
    "quant_artifact",
    "parallel_artifact",
    "fx_artifact",
    "runtime_event_artifact",
    "diagnostics_artifact",
    "replay_artifact",
    "microbench_artifact",
)
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


def _classify_file(repo_root: Path, current_file: str) -> tuple[str, str]:
    """Return the physical baseline layer and target responsibility layer."""
    parts = Path(current_file).parts
    physical = next((name for name in ("smoke", "regression", "benchmark") if name in parts), "non-layered")
    path = Path(current_file)
    normalized = path.as_posix()
    if normalized.startswith("tests/benchmark/models/"):
        return physical, "L1-model"
    if normalized.startswith("tests/benchmark/ops/"):
        return physical, "L2-module"
    in_refactor_scope = normalized.startswith(
        ("tests/regression/tensor_cast/", "tests/regression/cli/", "tests/regression/model_diagnostics/")
    )
    if not in_refactor_scope:
        return physical, "out-of-scope"
    try:
        source = (repo_root / path).read_text(encoding="utf-8")
    except OSError:
        return physical, "pending-review"
    if any(
        token in source for token in ("ModelRunner", "L1ScenarioExecutor", "model_id=", 'model_id = "', "model_id = '")
    ):
        return physical, "L1-model"
    l2_tokens = (
        "microbench",
        "op_replay",
        "build_op_registry",
        "RuntimeArtifactCapture",
        "diagnostics",
        "graph_extractor",
        "quantization",
    )
    if any(token in source for token in l2_tokens):
        return physical, "L2-module"
    return physical, "L3-unit"


def collect_inventory(args: argparse.Namespace) -> None:
    rows = []
    classifications: dict[str, tuple[str, str]] = {}
    for node_id in _node_ids(args.collect_output):
        current_file = node_id.split("::", 1)[0]
        current_layer, target_layer = classifications.setdefault(
            current_file, _classify_file(args.repo_root, current_file)
        )
        rows.append(
            {
                "node_id": node_id,
                "current_file": current_file,
                "current_layer": current_layer,
                "target_layer": target_layer,
                "target_file": current_file,
                "status": "master-baseline",
            }
        )
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
    incomplete = {
        family: sorted(ASSERTION_GROUPS - values) for family, values in groups.items() if values != ASSERTION_GROUPS
    }
    if incomplete:
        raise GateError(f"every family needs numeric/structural/capability assertions: {incomplete}")
    if args.baseline:
        baseline = _read_csv(args.baseline)
        _require_columns(args.baseline, baseline, required)
        baseline_active = {(r["family"], r["assertion_id"]) for r in baseline if r["status"] == "active"}
        waived = {
            (r["family"], r["assertion_id"]) for r in rows if r["status"] == "waived" and r["reason"] and r["approver"]
        }
        lost = sorted(baseline_active - active - waived)
        if lost:
            raise GateError(f"assertions disappeared without signed waiver: {lost}")
    if args.inventory and args.inventory.exists():
        inventory_rows = _read_csv(args.inventory)
        _require_columns(args.inventory, inventory_rows, INVENTORY_FIELDS)
        inventory_nodes = {row["node_id"] for row in inventory_rows}
        unknown = sorted(
            row["source_node"]
            for row in rows
            if row["status"] == "active" and row["source_node"] not in inventory_nodes
        )
        if unknown:
            raise GateError(f"active assertions reference nodes outside the master inventory: {unknown}")
    print(json.dumps({"families": len(groups), "active_assertions": len(active)}))


def check_scenarios(args: argparse.Namespace) -> None:
    required = (
        "family",
        "scenario_id",
        "build_signature",
        "run_signature",
        "build_count",
        "forward_count",
        "actual_seconds",
        "schedule",
    )
    rows = _read_csv(args.matrix)
    _require_columns(args.matrix, rows, required)
    invalid = sorted({row["schedule"] for row in rows} - {"incremental", "nightly", "pending-approval"})
    if invalid:
        raise GateError(f"invalid schedules: {invalid}")
    scenarios: dict[tuple[str, str], float] = {}
    builds = set()
    for line, row in enumerate((r for r in rows if r["schedule"] == "incremental"), 2):
        try:
            seconds = float(row["actual_seconds"])
            builds_for_run = int(row["build_count"])
            forwards_for_run = int(row["forward_count"])
        except ValueError as error:
            raise GateError(f"{args.matrix}:{line}: counts/actual_seconds must be numeric") from error
        if builds_for_run != 1 or forwards_for_run != 1:
            raise GateError(f"{args.matrix}:{line}: every measured scenario must build once and run once")
        key = (row["build_signature"], row["run_signature"])
        scenarios[key] = max(scenarios.get(key, 0.0), seconds)
        builds.add(row["build_signature"])
    total = sum(scenarios.values())
    print(
        json.dumps(
            {
                "builds": len(builds),
                "runs": len(scenarios),
                "expected_seconds": round(total, 3),
                "budget_seconds": args.budget,
            }
        )
    )
    if total > args.budget:
        raise GateError(f"incremental L1 scenario budget exceeded: {total:.3f}s > {args.budget:.3f}s")


def _load_test_map(path: Path) -> dict[str, dict[str, list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("map", payload)
    if not isinstance(mapping, dict):
        raise GateError(f"{path}: test map must contain an object-valued map")
    return mapping


def _test_map_commit(path: Path | None) -> str | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("built_from_commit")
    return value if isinstance(value, str) else None


def _symbol_watchers(mapping: dict[str, dict[str, list[str]]], product_symbol: str) -> set[str]:
    source_file, symbol = product_symbol.split("::", 1)
    return {node for node, sources in mapping.items() if symbol in sources.get(source_file, [])}


def check_watchers(args: argparse.Namespace) -> None:
    rows = _read_csv(args.watchers)
    required = ("family", "source_node", "product_symbol", "required")
    _require_columns(args.watchers, rows, required)
    mapping = _load_test_map(args.test_map)
    baseline_mapping = _load_test_map(args.baseline_test_map) if args.baseline_test_map else None
    selected_families = {"qwen", "dense", "moe", "multimodal"}
    if args.selection:
        selection_payload = json.loads(args.selection.read_text(encoding="utf-8"))
        selected_families = set(selection_payload.get("families", []))
        invalid = selected_families - {"qwen", "dense", "moe", "multimodal"}
        if invalid:
            raise GateError(f"invalid selected L1 families: {sorted(invalid)}")
    covered_families = set()
    required_watchers = 0
    missing = []
    missing_from_baseline = []
    family_evidence: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {
            "baseline_nodes": set(),
            "candidate_nodes": set(),
            "baseline_symbols": set(),
            "candidate_symbols": set(),
        }
    )
    for row in rows:
        if row["required"].lower() != "true" or row["family"] not in selected_families:
            continue
        required_watchers += 1
        source_file, symbol = row["product_symbol"].split("::", 1)
        symbols = mapping.get(row["source_node"], {}).get(source_file, [])
        if symbol not in symbols:
            missing.append(f"{row['family']}: {row['source_node']} -> {row['product_symbol']}")
        else:
            family_evidence[row["family"]]["candidate_nodes"].add(row["source_node"])
            family_evidence[row["family"]]["candidate_symbols"].add(row["product_symbol"])
        if baseline_mapping is not None:
            baseline_nodes = _symbol_watchers(baseline_mapping, row["product_symbol"])
            if not baseline_nodes:
                missing_from_baseline.append(f"{row['family']}: {row['product_symbol']}")
            else:
                family_evidence[row["family"]]["baseline_nodes"].update(baseline_nodes)
                family_evidence[row["family"]]["baseline_symbols"].add(row["product_symbol"])
        covered_families.add(row["family"])
    if covered_families != selected_families:
        raise GateError(f"watcher manifest must cover selected families: {sorted(selected_families)}")
    if missing:
        raise GateError("required watcher paths disappeared:\n" + "\n".join(missing))
    if missing_from_baseline:
        raise GateError(
            "required symbols have no watcher in the baseline test map:\n" + "\n".join(missing_from_baseline)
        )
    evidence = []
    lost_symbols = []
    for family in sorted(covered_families):
        item = family_evidence[family]
        lost = item["baseline_symbols"] - item["candidate_symbols"]
        lost_symbols.extend(f"{family}: {symbol}" for symbol in sorted(lost))
        evidence.append(
            {
                "family": family,
                "baseline_nodes": sorted(item["baseline_nodes"]),
                "candidate_nodes": sorted(item["candidate_nodes"]),
                "baseline_symbols": sorted(item["baseline_symbols"]),
                "candidate_symbols": sorted(item["candidate_symbols"]),
                "lost_symbols": sorted(lost),
            }
        )
    if lost_symbols:
        raise GateError("candidate watcher coverage lost baseline symbols:\n" + "\n".join(lost_symbols))
    if args.evidence_output:
        args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "coverage_context": "test",
                    "baseline_commit": _test_map_commit(args.baseline_test_map),
                    "candidate_commit": _test_map_commit(args.test_map),
                    "baseline_test_map": str(args.baseline_test_map) if args.baseline_test_map else None,
                    "candidate_test_map": str(args.test_map),
                    "families": evidence,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "families": len(covered_families),
                "required_watchers": required_watchers,
                "conserved_symbols": sum(len(item["candidate_symbols"]) for item in family_evidence.values()),
            }
        )
    )


def check_family_batches(args: argparse.Namespace) -> None:
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    families = registry.get("families", {})
    scenarios = _read_csv(args.scenarios)
    assertions = _read_csv(args.assertions)
    watchers = _read_csv(args.watchers)
    expected_families = {"qwen", "dense", "moe", "multimodal"}
    if set(families) != expected_families:
        raise GateError("family batch registry must contain exactly qwen/dense/moe/multimodal")
    orders = set()
    errors = []
    for family, record in families.items():
        order = record.get("order")
        orders.add(order)
        evidence = record.get("evidence", {})
        actual = {
            "scenarios": sum(row["family"] == family for row in scenarios),
            "assertions": sum(row["family"] == family for row in assertions),
            "watchers": sum(row["family"] == family and row["required"].lower() == "true" for row in watchers),
        }
        if evidence.get("status") != "passed":
            errors.append(f"{family}: batch evidence status must be passed")
        for key, count in actual.items():
            if evidence.get(key) != count:
                errors.append(f"{family}: evidence {key}={evidence.get(key)!r}, actual={count}")
        rollback = record.get("rollback", {})
        if not rollback.get("selector") or not rollback.get("scope"):
            errors.append(f"{family}: rollback selector and scope are required")
    if orders != {1, 2, 3, 4}:
        errors.append(f"family batch order must be 1..4, got {sorted(orders, key=str)}")
    if errors:
        raise GateError("\n".join(errors))
    print(json.dumps({"families": 4, "batch_order": ["qwen", "dense", "moe", "multimodal"]}))


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


def _check_cov_helper(path: Path, source: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    errors = []
    decorators = {_name(item) for item in node.decorator_list}
    is_cov_helper = any(name.endswith("cov_helper") for name in decorators)
    docstring = ast.get_docstring(node) or ""
    coverage_only = "line coverage only" in docstring.lower()
    if coverage_only and not is_cov_helper:
        errors.append(f"{path}:{node.lineno}: coverage-only L3 test must use @pytest.mark.cov_helper")
    if not is_cov_helper:
        return errors
    segment = ast.get_source_segment(source, node) or ""
    marker_lines = [line.strip() for line in docstring.splitlines() if line.strip().startswith("COV_HELPER:")]
    if len(marker_lines) != 1 or "::" not in marker_lines[0]:
        errors.append(f"{path}:{node.lineno}: cov_helper needs one COV_HELPER: product/path.py::symbol line")
    if any(token in segment for token in ('patch("tensor_cast.', 'patch("cli.', 'patch("serving_cast.')):
        errors.append(f"{path}:{node.lineno}: cov_helper must not mock the product object under test")
    return errors


def check_cov_helpers(_args: argparse.Namespace) -> None:
    errors = []
    marked = 0
    files = sorted(Path("tests").glob("**/unit/test_*.py"))
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
                continue
            decorators = {_name(item) for item in node.decorator_list}
            marked += any(name.endswith("cov_helper") for name in decorators)
            errors.extend(_check_cov_helper(path, source, node))
    if errors:
        raise GateError("\n".join(errors))
    print(json.dumps({"checked_l3_files": len(files), "cov_helpers": marked}))


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
        execution_exempt = "L1_EXECUTION_EXEMPT:" in source
        if (
            uses_model_execution
            and not uses_executor
            and not execution_exempt
            and "tests/helpers/tests/" not in path.as_posix()
        ):
            errors.append(f"{path}: L1 model execution must use L1ScenarioExecutor/l1_executor")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
                continue
            errors.extend(_check_cov_helper(path, source, node))
        if (
            uses_model_execution
            and not execution_exempt
            and registered
            and not any(f"L1_FAMILY: {family}" in source for family in registered)
        ):
            errors.append(f"{path}: L1 test must declare a registered 'L1_FAMILY: <name>'")
    if errors:
        raise GateError("\n".join(errors))
    print(json.dumps({"checked_files": len(files)}))


def check_l2_layers(args: argparse.Namespace) -> None:
    """Reject real model construction from every L2 helper and module test."""
    helper = Path("tests/helpers/l2_artifacts.py")
    module_files = sorted(Path("tests/regression").glob("*/module/**/*.py"))
    files = [helper, *module_files]
    violations = [violation for path in files if path.is_file() for violation in find_l2_model_violations_in_file(path)]
    if violations:
        raise GateError("L2 model-construction violations:\n" + "\n".join(violations))
    consumers = {
        builder: sorted(
            path.as_posix()
            for path in module_files
            if path.name != "test_l2_artifacts.py" and f"{builder}(" in path.read_text(encoding="utf-8")
        )
        for builder in L2_BUILDERS
    }
    if helper.is_file() and "L2_CONTRACTS" in helper.read_text(encoding="utf-8"):
        missing = [builder for builder, paths in consumers.items() if not paths]
        if missing:
            raise GateError(f"canonical L2 builders lack real module consumers: {', '.join(missing)}")
    print(
        json.dumps(
            {
                "checked_l2_files": len(files),
                "violations": 0,
                "canonical_builder_consumers": {builder: len(paths) for builder, paths in consumers.items()},
            }
        )
    )


def check_test_policy(args: argparse.Namespace) -> None:
    """Run every static test-policy rule while reporting all violations."""
    checks = (
        (
            "assertions",
            check_assertions,
            argparse.Namespace(
                manifest=args.assertion_manifest,
                baseline=args.assertion_baseline,
                inventory=args.inventory,
            ),
        ),
        (
            "scenarios",
            check_scenarios,
            argparse.Namespace(matrix=args.scenario_matrix, budget=args.budget),
        ),
        (
            "family-batches",
            check_family_batches,
            argparse.Namespace(
                registry=args.family_registry,
                scenarios=args.scenario_matrix,
                assertions=args.assertion_manifest,
                watchers=args.watchers,
            ),
        ),
        ("l2-layers", check_l2_layers, argparse.Namespace()),
        ("cov-helpers", check_cov_helpers, argparse.Namespace()),
        (
            "new-tests",
            check_new_tests,
            argparse.Namespace(base=args.base, family_registry=args.family_registry),
        ),
    )
    failures = []
    for name, check, check_args in checks:
        try:
            check(check_args)
        except (GateError, OSError, subprocess.CalledProcessError) as error:
            failures.append(f"{name}: {error}")
    if failures:
        raise GateError("test policy violations:\n" + "\n".join(failures))
    print(json.dumps({"stage": "check-test-policy", "checks": len(checks), "status": "passed"}))


def _add_watcher_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--watchers", type=Path, default=Path("tests/.ci/l1_watchers.csv"))
    parser.add_argument("--test-map", type=Path, default=Path("artifacts/l1_test_map.json"))
    parser.add_argument("--baseline-test-map", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--evidence-output", type=Path, default=Path("artifacts/l1_watcher_conservation.json"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect-inventory")
    collect.add_argument("--collect-output", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--repo-root", type=Path, default=Path.cwd())
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
    watchers = sub.add_parser("check-watchers")
    _add_watcher_arguments(watchers)
    watchers.set_defaults(func=check_watchers)
    conservation = sub.add_parser("check-test-conservation")
    _add_watcher_arguments(conservation)
    conservation.set_defaults(func=check_watchers)
    batches = sub.add_parser("check-family-batches")
    batches.add_argument("--registry", type=Path, default=Path("tests/.ci/l1_families.json"))
    batches.add_argument("--scenarios", type=Path, default=Path("tests/.ci/l1_scenarios.csv"))
    batches.add_argument("--assertions", type=Path, default=Path("tests/.ci/assertion_manifest.csv"))
    batches.add_argument("--watchers", type=Path, default=Path("tests/.ci/l1_watchers.csv"))
    batches.set_defaults(func=check_family_batches)
    new_tests = sub.add_parser("check-new-tests")
    new_tests.add_argument("--base", default="origin/master")
    new_tests.add_argument("--family-registry", type=Path, default=Path("tests/.ci/l1_families.json"))
    new_tests.set_defaults(func=check_new_tests)
    layers = sub.add_parser("check-l2-layers")
    layers.set_defaults(func=check_l2_layers)
    cov_helpers = sub.add_parser("check-cov-helpers")
    cov_helpers.set_defaults(func=check_cov_helpers)
    policy = sub.add_parser("check-test-policy")
    policy.add_argument("--base", default="origin/master")
    policy.add_argument("--budget", type=float, default=180.0)
    policy.add_argument("--assertion-manifest", type=Path, default=Path("tests/.ci/assertion_manifest.csv"))
    policy.add_argument("--assertion-baseline", type=Path)
    policy.add_argument("--inventory", type=Path, default=Path("tests/.ci/inventory/master_test_inventory.csv"))
    policy.add_argument("--scenario-matrix", type=Path, default=Path("tests/.ci/l1_scenarios.csv"))
    policy.add_argument("--family-registry", type=Path, default=Path("tests/.ci/l1_families.json"))
    policy.add_argument("--watchers", type=Path, default=Path("tests/.ci/l1_watchers.csv"))
    policy.set_defaults(func=check_test_policy)
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

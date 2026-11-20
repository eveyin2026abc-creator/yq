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
from collections.abc import Iterable
from pathlib import Path

from tests.architecture.layer_guard import find_l2_model_violations_in_file

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
    if errors:
        raise GateError("\n".join(errors))
    print(json.dumps({"checked_files": len(files)}))


def _attach_ast_parents(tree: ast.AST) -> None:
    """Annotate an AST so a builder call can be checked in its statement context."""
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent


def _l2_builder_usage(path: Path) -> tuple[set[str], list[str]]:
    """Return consumed canonical builders and discarded-call violations."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    _attach_ast_parents(tree)
    builders: set[str] = set()
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        builder = _name(node.func)
        if builder not in L2_BUILDERS:
            continue
        builders.add(builder)
        if isinstance(node._parent, ast.Expr):
            violations.append(f"{path}:{node.lineno}: discards {builder} result")
    return builders, violations


def _l2_exemption_reason(path: Path) -> str:
    """Read a file-level or enclosing module-level canonical-artifact exemption."""
    candidates = [path]
    parent = path.parent
    while "module" in parent.parts:
        candidates.append(parent / "__init__.py")
        if parent.name == "module":
            break
        parent = parent.parent
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            if "L2_ARTIFACT_EXEMPT:" in line:
                return line.split("L2_ARTIFACT_EXEMPT:", 1)[1].strip()
    return ""


def check_l2_layers(args: argparse.Namespace) -> None:
    """Reject real model construction from every L2 helper and module test."""
    helper = Path("tests/helpers/l2_artifacts.py")
    module_files = sorted(Path("tests/regression").glob("*/module/**/*.py"))
    files = [helper, *module_files]
    violations = [violation for path in files if path.is_file() for violation in find_l2_model_violations_in_file(path)]
    if violations:
        raise GateError("L2 model-construction violations:\n" + "\n".join(violations))
    usage = {
        path: _l2_builder_usage(path)
        for path in module_files
        if path.name.startswith("test") and path.name != "test_l2_artifacts.py"
    }
    discarded = [violation for _, path_violations in usage.values() for violation in path_violations]
    if discarded:
        raise GateError("discarded canonical L2 artifacts:\n" + "\n".join(discarded))
    consumers = {
        builder: sorted(path.as_posix() for path, (builders, _) in usage.items() if builder in builders)
        for builder in L2_BUILDERS
    }
    exemptions: dict[str, str] = {}
    unclassified: list[str] = []
    for path, (builders, _) in usage.items():
        if builders:
            continue
        marker = _l2_exemption_reason(path)
        if len(marker) < 20:
            unclassified.append(f"{path}: use a canonical builder or add a specific L2_ARTIFACT_EXEMPT reason")
        else:
            exemptions[path.as_posix()] = marker
    if unclassified:
        raise GateError("unclassified L2 module tests:\n" + "\n".join(unclassified))
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
                "explicit_exemptions": len(exemptions),
            }
        )
    )


def check_test_policy(args: argparse.Namespace) -> None:
    """Run every static test-policy rule while reporting all violations."""
    checks = (
        ("l2-layers", check_l2_layers, argparse.Namespace()),
        ("cov-helpers", check_cov_helpers, argparse.Namespace()),
        (
            "new-tests",
            check_new_tests,
            argparse.Namespace(base=args.base),
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
    new_tests = sub.add_parser("check-new-tests")
    new_tests.add_argument("--base", default="origin/master")
    new_tests.set_defaults(func=check_new_tests)
    layers = sub.add_parser("check-l2-layers")
    layers.set_defaults(func=check_l2_layers)
    cov_helpers = sub.add_parser("check-cov-helpers")
    cov_helpers.set_defaults(func=check_cov_helpers)
    policy = sub.add_parser("check-test-policy")
    policy.add_argument("--base", default="origin/master")
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

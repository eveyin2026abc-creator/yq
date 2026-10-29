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
"""Regression tests for the test architecture mechanical gates."""

import argparse
import ast
import csv
import json
from pathlib import Path

import pytest

from scripts.helpers.test_architecture import (
    GateError,
    _changed_test_files,
    _classify_file,
    _name,
    build_parser,
    check_assertions,
    check_new_tests,
    check_scenarios,
    check_watchers,
    compare_inventory,
    collect_inventory,
    main,
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def test_collect_inventory_uses_exact_node_ids(tmp_path: Path) -> None:
    collected = tmp_path / "collected.txt"
    collected.write_text(
        "tests/a/test_x.py::test_one\n"
        "tests/a/test_x.py::test_param[value with spaces]\n"
        "2 tests collected\n",
        encoding="utf-8",
    )
    output = tmp_path / "inventory.csv"
    collect_inventory(argparse.Namespace(collect_output=collected, output=output, repo_root=tmp_path))
    assert "tests/a/test_x.py::test_one" in output.read_text(encoding="utf-8")
    assert "tests/a/test_x.py::test_param[value with spaces]" in output.read_text(encoding="utf-8")


def test_classify_file_routes_model_and_pure_tests(tmp_path: Path) -> None:
    model = tmp_path / "tests/regression/tensor_cast/test_model.py"
    model.parent.mkdir(parents=True)
    model.write_text("runner = ModelRunner(model_id=\"Qwen/Qwen3\")\n", encoding="utf-8")
    unit = tmp_path / "tests/regression/cli/test_parser.py"
    unit.parent.mkdir(parents=True)
    unit.write_text("def test_parse():\n    assert int('1') == 1\n", encoding="utf-8")

    assert _classify_file(tmp_path, model.relative_to(tmp_path).as_posix()) == ("regression", "L1-model")
    assert _classify_file(tmp_path, unit.relative_to(tmp_path).as_posix()) == ("regression", "L3-unit")


def test_assertion_manifest_requires_all_three_groups(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    fields = ("family", "group", "assertion_id", "source_node", "product_symbol", "status", "reason", "approver")
    _write_csv(manifest, fields, [("qwen", "numeric", "n1", "tests/a.py::test_a", "p::s", "active", "", "")])
    with pytest.raises(GateError, match="numeric/structural/capability"):
        check_assertions(argparse.Namespace(manifest=manifest, baseline=None, inventory=None))


def test_assertion_manifest_requires_signed_waiver(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    fields = ("family", "group", "assertion_id", "source_node", "product_symbol", "status", "reason", "approver")
    rows = [("qwen", group, group, f"tests/a.py::test_{group}", "p::s", "waived" if group == "numeric" else "active", "", "") for group in ("numeric", "structural", "capability")]
    _write_csv(manifest, fields, rows)
    with pytest.raises(GateError, match="requires reason and approver"):
        check_assertions(argparse.Namespace(manifest=manifest, baseline=None, inventory=None))


def test_scenario_budget_counts_unique_build_run_pairs(tmp_path: Path) -> None:
    matrix = tmp_path / "scenarios.csv"
    fields = ("family", "scenario_id", "build_signature", "run_signature", "build_count", "forward_count", "actual_seconds", "schedule")
    rows = [
        ("qwen", "numeric", "build-a", "run-a", "1", "1", "90", "incremental"),
        ("qwen", "structural", "build-a", "run-a", "1", "1", "90", "incremental"),
    ]
    _write_csv(matrix, fields, rows)
    check_scenarios(argparse.Namespace(matrix=matrix, budget=100.0))


def test_scenario_budget_rejects_too_many_distinct_runs(tmp_path: Path) -> None:
    matrix = tmp_path / "scenarios.csv"
    fields = ("family", "scenario_id", "build_signature", "run_signature", "build_count", "forward_count", "actual_seconds", "schedule")
    rows = [
        ("qwen", "one", "build-a", "run-a", "1", "1", "100", "incremental"),
        ("qwen", "two", "build-a", "run-b", "1", "1", "100", "incremental"),
    ]
    _write_csv(matrix, fields, rows)
    with pytest.raises(GateError, match="budget exceeded"):
        check_scenarios(argparse.Namespace(matrix=matrix, budget=180.0))


def test_watcher_gate_requires_mapped_symbol(tmp_path: Path) -> None:
    watchers = tmp_path / "watchers.csv"
    fields = ("family", "source_node", "product_symbol", "required")
    rows = [(family, f"tests/a.py::test_{family}", f"product/{family}.py::run", "true") for family in ("qwen", "dense", "moe", "multimodal")]
    _write_csv(watchers, fields, rows)
    test_map = tmp_path / "test_map.json"
    test_map.write_text(json.dumps({"map": {row[1]: {f"product/{row[0]}.py": ["run"]} for row in rows}}), encoding="utf-8")
    check_watchers(argparse.Namespace(watchers=watchers, test_map=test_map))


def test_compare_inventory_writes_empty_reports_for_matching_nodes(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.csv"
    fields = ("node_id", "current_file", "current_layer", "target_layer", "target_file", "status")
    node = "tests/a.py::test_one"
    _write_csv(inventory, fields, [(node, "tests/a.py", "regression", "L3-unit", "tests/a.py", "master-baseline")])
    collected = tmp_path / "collected.txt"
    collected.write_text(node + "\n", encoding="utf-8")

    compare_inventory(argparse.Namespace(inventory=inventory, collect_output=collected, output_dir=tmp_path / "reports"))

    assert (tmp_path / "reports/inventory_missing.csv").read_text(encoding="utf-8") == "node_id\n"
    assert (tmp_path / "reports/inventory_extra.csv").read_text(encoding="utf-8") == "node_id\n"


def test_changed_test_files_keeps_only_added_existing_python_tests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_file = tmp_path / "tests/new/test_case.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_case(): pass\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "scripts.helpers.test_architecture.subprocess.run",
        lambda *_args, **_kwargs: argparse.Namespace(stdout="tests/new/test_case.py\ntests/new/readme.md\ntests/missing.py\n"),
    )

    assert _changed_test_files("origin/master") == [Path("tests/new/test_case.py")]


def test_name_formats_nested_ast_names() -> None:
    assert _name(ast.parse("pkg.Factory().run").body[0].value) == "pkg.Factory.run"
    assert _name(ast.Constant(value=1)) == ""


def test_check_new_tests_accepts_registered_executor_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_file = tmp_path / "tests/regression/tensor_cast/model/test_new.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "# L1_FAMILY: qwen\n"
        "from tests.helpers.l1_scenario import L1ScenarioExecutor\n"
        "def test_new():\n    assert L1ScenarioExecutor\n",
        encoding="utf-8",
    )
    registry = tmp_path / "families.json"
    registry.write_text(json.dumps({"families": {"qwen": {}}}), encoding="utf-8")
    monkeypatch.setattr("scripts.helpers.test_architecture._changed_test_files", lambda _base: [test_file])

    check_new_tests(argparse.Namespace(base="origin/master", family_registry=registry))


def test_build_parser_exposes_all_architecture_commands() -> None:
    parser = build_parser()
    choices = next(action for action in parser._actions if action.dest == "command").choices
    assert set(choices) == {
        "collect-inventory",
        "compare-inventory",
        "check-assertions",
        "check-scenarios",
        "check-watchers",
        "check-new-tests",
    }


def test_main_reports_gate_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    parser = argparse.Namespace(
        parse_args=lambda: argparse.Namespace(
            func=lambda _args: (_ for _ in ()).throw(GateError("boom"))
        )
    )
    monkeypatch.setattr("scripts.helpers.test_architecture.build_parser", lambda: parser)

    assert main() == 1
    assert "test architecture gate: boom" in capsys.readouterr().err

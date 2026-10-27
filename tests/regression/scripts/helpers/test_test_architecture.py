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
import csv
from pathlib import Path

import pytest

from scripts.helpers.test_architecture import GateError, check_assertions, check_scenarios, collect_inventory


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
    collect_inventory(argparse.Namespace(collect_output=collected, output=output))
    assert "tests/a/test_x.py::test_one" in output.read_text(encoding="utf-8")
    assert "tests/a/test_x.py::test_param[value with spaces]" in output.read_text(encoding="utf-8")


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
    fields = ("family", "scenario_id", "build_signature", "run_signature", "expected_seconds", "schedule")
    rows = [
        ("qwen", "numeric", "build-a", "run-a", "90", "incremental"),
        ("qwen", "structural", "build-a", "run-a", "90", "incremental"),
    ]
    _write_csv(matrix, fields, rows)
    check_scenarios(argparse.Namespace(matrix=matrix, budget=100.0))


def test_scenario_budget_rejects_too_many_distinct_runs(tmp_path: Path) -> None:
    matrix = tmp_path / "scenarios.csv"
    fields = ("family", "scenario_id", "build_signature", "run_signature", "expected_seconds", "schedule")
    rows = [
        ("qwen", "one", "build-a", "run-a", "100", "incremental"),
        ("qwen", "two", "build-a", "run-b", "100", "incremental"),
    ]
    _write_csv(matrix, fields, rows)
    with pytest.raises(GateError, match="budget exceeded"):
        check_scenarios(argparse.Namespace(matrix=matrix, budget=180.0))

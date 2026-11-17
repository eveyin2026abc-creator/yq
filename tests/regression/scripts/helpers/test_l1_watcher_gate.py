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
"""Tests for dynamically generated L1 watcher evidence."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.helpers import l1_watcher_gate


def _write_watchers(path: Path, *, omit: str | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("family", "source_node", "product_symbol", "required"))
        for family in ("qwen", "dense", "moe", "multimodal"):
            if family != omit:
                writer.writerow((family, f"tests/l1.py::test_{family}", f"product/{family}.py::run", "true"))


def test_required_nodes_returns_all_four_families(tmp_path: Path) -> None:
    watchers = tmp_path / "watchers.csv"
    _write_watchers(watchers)
    assert set(l1_watcher_gate.required_nodes(watchers)) == {
        "tests/l1.py::test_qwen",
        "tests/l1.py::test_dense",
        "tests/l1.py::test_moe",
        "tests/l1.py::test_multimodal",
    }


def test_required_nodes_rejects_incomplete_contract(tmp_path: Path) -> None:
    watchers = tmp_path / "watchers.csv"
    _write_watchers(watchers, omit="moe")
    with pytest.raises(ValueError, match="moe"):
        l1_watcher_gate.required_nodes(watchers)


def test_generate_candidate_runs_non_nightly_and_writes_observed_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watchers = tmp_path / "watchers.csv"
    _write_watchers(watchers)
    output = tmp_path / "artifacts/l1_test_map.json"
    coverage = tmp_path / ".coverage.l1"
    observed = {"tests/l1.py::test_qwen": {"product/qwen.py": ["run"]}}
    calls: dict[str, object] = {}

    def fake_run(command, *, cwd, env, check):
        calls.update(command=command, cwd=cwd, env=env, check=check)
        coverage.write_text("coverage", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        l1_watcher_gate,
        "pytest_command",
        lambda nodes: ["python", "-m", "pytest", *nodes, "-m", "not nightly", "--cov-context=test"],
    )
    monkeypatch.setattr(l1_watcher_gate.subprocess, "run", fake_run)
    monkeypatch.setattr(l1_watcher_gate, "collect_from_coverage", lambda *_args, **_kwargs: observed)
    monkeypatch.setattr(
        l1_watcher_gate,
        "write_test_map",
        lambda path, mapping: calls.update(output=path, mapping=mapping),
    )

    assert l1_watcher_gate.generate_candidate(watchers, output, coverage) == 0
    assert calls["mapping"] == observed
    assert calls["output"] == output
    assert calls["env"]["COVERAGE_FILE"] == str(coverage.resolve())
    command = calls["command"]
    assert "--cov-context=test" in command
    assert command[command.index("-m", 3) + 1] == "not nightly"

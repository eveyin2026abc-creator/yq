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
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.helpers.architecture import l1_watchers as l1_watcher_gate


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


def test_required_nodes_reads_the_authoritative_contract_registry() -> None:
    nodes = l1_watcher_gate.required_nodes(Path("tests/.ci/l1_contracts.json"))

    assert nodes == (
        "tests/regression/tensor_cast/model/test_l1_family_scenarios.py::test_dense_scenarios",
        "tests/regression/tensor_cast/model/test_l1_family_scenarios.py::test_moe_scenarios",
        "tests/regression/tensor_cast/model/test_l1_family_scenarios.py::test_multimodal_scenarios",
        "tests/regression/tensor_cast/model/test_l1_family_scenarios.py::test_qwen_scenarios",
    )


def test_required_nodes_rejects_incomplete_contract(tmp_path: Path) -> None:
    watchers = tmp_path / "watchers.csv"
    _write_watchers(watchers, omit="moe")
    with pytest.raises(ValueError, match="moe"):
        l1_watcher_gate.required_nodes(watchers)


def test_select_families_skips_unrelated_change(tmp_path: Path) -> None:
    watchers = tmp_path / "watchers.csv"
    _write_watchers(watchers)
    families, reasons = l1_watcher_gate.select_families(watchers, ("docs/guide.md",))
    assert families == frozenset()
    assert reasons == {}


def test_select_families_uses_watched_product_paths(tmp_path: Path) -> None:
    watchers = tmp_path / "watchers.csv"
    _write_watchers(watchers)
    families, reasons = l1_watcher_gate.select_families(watchers, ("product/moe.py",))
    assert families == frozenset({"moe"})
    assert reasons == {"moe": ["product/moe.py"]}


def test_select_families_runs_all_for_shared_l1_infrastructure(tmp_path: Path) -> None:
    watchers = tmp_path / "watchers.csv"
    _write_watchers(watchers)
    families, _reasons = l1_watcher_gate.select_families(watchers, ("tests/helpers/l1_scenario.py",))
    assert families == l1_watcher_gate._FAMILIES


def test_pytest_command_selects_non_nightly_nodes() -> None:
    command = l1_watcher_gate.pytest_command(("tests/l1.py::test_qwen",))
    assert command[:3] == [l1_watcher_gate.sys.executable, "-m", "pytest"]
    assert "tests/l1.py::test_qwen" in command
    assert command[command.index("-m", 3) + 1] == "not nightly"
    assert "--cov-context=test" in command


def test_build_parser_accepts_all_artifact_paths() -> None:
    args = l1_watcher_gate.build_parser().parse_args(
        [
            "--watchers",
            "watchers.csv",
            "--output",
            "candidate.json",
            "--coverage-file",
            "coverage.data",
            "--changed-paths-file",
            "changed.txt",
            "--selection-output",
            "selection.json",
        ]
    )
    assert args.watchers == Path("watchers.csv")
    assert args.output == Path("candidate.json")
    assert args.coverage_file == Path("coverage.data")
    assert args.changed_paths_file == Path("changed.txt")
    assert args.selection_output == Path("selection.json")


def test_main_routes_paths_to_candidate_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    watchers = tmp_path / "watchers.csv"
    _write_watchers(watchers)
    output = tmp_path / "candidate.json"
    coverage = tmp_path / "coverage.data"
    changed = tmp_path / "changed.txt"
    changed.write_text("tests/helpers/l1_scenario.py\n", encoding="utf-8")
    selection = tmp_path / "selection.json"
    parsed = SimpleNamespace(
        watchers=watchers,
        output=output,
        coverage_file=coverage,
        changed_paths_file=changed,
        selection_output=selection,
    )
    monkeypatch.setattr(l1_watcher_gate, "build_parser", lambda: SimpleNamespace(parse_args=lambda: parsed))
    calls: list[tuple[Path, Path, Path]] = []
    monkeypatch.setattr(
        l1_watcher_gate,
        "generate_candidate",
        lambda watchers_path, output_path, coverage_path, families: calls.append(
            (watchers_path, output_path, coverage_path, families)
        )
        or 0,
    )
    assert l1_watcher_gate.main() == 0
    assert calls == [(watchers, output, coverage, l1_watcher_gate._FAMILIES)]
    assert set(json.loads(selection.read_text(encoding="utf-8"))["families"]) == set(l1_watcher_gate._FAMILIES)


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


def test_generate_candidate_skips_pytest_when_no_family_is_affected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watchers = tmp_path / "watchers.csv"
    _write_watchers(watchers)
    output = tmp_path / "candidate.json"
    coverage = tmp_path / ".coverage.l1"
    written = {}
    monkeypatch.setattr(
        l1_watcher_gate,
        "write_test_map",
        lambda path, mapping: written.update(path=path, mapping=mapping),
    )
    monkeypatch.setattr(
        l1_watcher_gate.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("pytest must not run for an unrelated PR"),
    )
    assert l1_watcher_gate.generate_candidate(watchers, output, coverage, frozenset()) == 0
    assert written == {"path": output, "mapping": {}}

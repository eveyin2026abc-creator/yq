"""Tests for immediate pytest failure journaling."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

from scripts.helpers.nightly import failure_journal

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _failed_report() -> SimpleNamespace:
    return SimpleNamespace(
        failed=True,
        nodeid="tests/a.py::test_broken",
        outcome="failed",
        when="call",
        duration=1.25,
        longreprtext="AssertionError: boom",
        longrepr="AssertionError: boom",
    )


def test_serial_failure_is_flushed_to_jsonl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    journal = tmp_path / "failures.jsonl"
    monkeypatch.setenv("MSMODELING_NIGHTLY_FAILURE_JOURNAL", str(journal))
    config = SimpleNamespace(option=SimpleNamespace(numprocesses=None))

    failure_journal.pytest_configure(config)
    failure_journal.pytest_runtest_logreport(_failed_report())

    event = json.loads(journal.read_text(encoding="utf-8"))
    assert event["worker"] == "master"
    assert event["node_id"] == "tests/a.py::test_broken"
    assert event["phase"] == "call"
    assert event["traceback"] == "AssertionError: boom"


def test_xdist_controller_skips_and_worker_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    journal = tmp_path / "failures.jsonl"
    monkeypatch.setenv("MSMODELING_NIGHTLY_FAILURE_JOURNAL", str(journal))

    controller = SimpleNamespace(option=SimpleNamespace(numprocesses=8))
    failure_journal.pytest_configure(controller)
    failure_journal.pytest_runtest_logreport(_failed_report())
    assert not journal.exists()

    worker = SimpleNamespace(
        option=SimpleNamespace(numprocesses=8),
        workerinput={"workerid": "gw3"},
    )
    failure_journal.pytest_configure(worker)
    failure_journal.pytest_runtest_logreport(_failed_report())
    event = json.loads(journal.read_text(encoding="utf-8"))
    assert event["worker"] == "gw3"

"""Tests for duration-aware nightly slow-first scheduling."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

from scripts.helpers.nightly import slow_first
from scripts.helpers.nightly.slow_first import (
    HISTORY_REL,
    SAMPLES_REL,
    load_history,
    merge_previous_samples,
    ranked_slow_nodeids,
    resolve_slow_first_count,
    spread_slow_indices,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_rank_uses_median_of_latest_five_and_requires_three_samples() -> None:
    history = {
        "tests/a.py::stable": [8.0, 10.0, 9.0, 11.0, 100.0, 12.0],
        "tests/a.py::slower": [20.0, 18.0, 19.0],
        "tests/a.py::too_new": [100.0, 100.0],
    }
    assert ranked_slow_nodeids(history, 16) == [
        "tests/a.py::slower",
        "tests/a.py::stable",
    ]


def test_slow_count_supports_zero_eight_and_sixteen(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("0", "8", "16"):
        monkeypatch.setenv("MSMODELING_NIGHTLY_SLOW_FIRST", value)
        assert resolve_slow_first_count() == int(value)
    monkeypatch.setenv("MSMODELING_NIGHTLY_SLOW_FIRST", "invalid")
    assert resolve_slow_first_count() == 16


def test_spread_assigns_top16_to_distinct_workers() -> None:
    collection = [f"tests/a.py::test_{index}" for index in range(20)]
    ranked = collection[:16]
    assignments = spread_slow_indices(collection, ranked, worker_count=128)
    nonempty = [indices for indices in assignments if indices]
    assert len(nonempty) == 16
    assert all(len(indices) == 1 for indices in nonempty)
    assert [indices[0] for indices in nonempty] == list(range(16))


def test_merge_keeps_latest_five_samples_per_node(tmp_path: Path) -> None:
    history_path = tmp_path / HISTORY_REL
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        json.dumps({"tests/a.py::test_slow": [1, 2, 3, 4, 5]}),
        encoding="utf-8",
    )
    samples_path = tmp_path / SAMPLES_REL
    samples_path.write_text(
        "\n".join(
            [
                json.dumps({"node_id": "tests/a.py::test_slow", "duration_sec": 6}),
                json.dumps({"node_id": "tests/a.py::test_new", "duration_sec": 7}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    merged = merge_previous_samples(tmp_path)
    assert merged["tests/a.py::test_slow"] == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert merged["tests/a.py::test_new"] == [7.0]
    assert load_history(history_path) == merged
    assert not samples_path.exists()


def test_controller_records_call_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / HISTORY_REL
    monkeypatch.setenv("MSMODELING_NIGHTLY_SLOW_HISTORY_PATH", str(history_path))
    slow_first.pytest_configure(
        SimpleNamespace(option=SimpleNamespace(numprocesses=8))
    )
    slow_first.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid="tests/a.py::test_slow",
            when="call",
            duration=12.5,
        )
    )
    event = json.loads((tmp_path / SAMPLES_REL).read_text(encoding="utf-8"))
    assert event == {
        "node_id": "tests/a.py::test_slow",
        "duration_sec": 12.5,
    }

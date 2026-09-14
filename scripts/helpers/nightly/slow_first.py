"""Duration-aware xdist scheduler for nightly Wave A.

The scheduler ranks tests by the median of their latest five call durations.
At least three samples are required. Selected slow tests are assigned to
different workers before regular work is distributed by xdist worksteal.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from pathlib import Path
from typing import Any, Final

from xdist.scheduler.worksteal import WorkStealingScheduling

_COUNT_ENV: Final = "MSMODELING_NIGHTLY_SLOW_FIRST"
_HISTORY_ENV: Final = "MSMODELING_NIGHTLY_SLOW_HISTORY_PATH"
_DEFAULT_COUNT: Final = 16
_MAX_SAMPLES: Final = 5
_MIN_SAMPLES: Final = 3
HISTORY_REL: Final = ".pytest_cache/nightly/slow-duration-history.json"
SAMPLES_REL: Final = ".pytest_cache/nightly/slow-duration-samples.jsonl"

_sample_path: Path | None = None
_write_samples = False


def resolve_slow_first_count() -> int:
    raw = (os.environ.get(_COUNT_ENV) or "").strip()
    if not raw:
        return _DEFAULT_COUNT
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_COUNT


def load_history(path: Path) -> dict[str, list[float]]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    result: dict[str, list[float]] = {}
    for node_id, values in raw.items():
        if not isinstance(values, list):
            continue
        samples: list[float] = []
        for value in values[-_MAX_SAMPLES:]:
            try:
                duration = float(value)
            except (TypeError, ValueError):
                continue
            if duration >= 0:
                samples.append(duration)
        if samples:
            result[str(node_id)] = samples
    return result


def ranked_slow_nodeids(history: dict[str, list[float]], count: int) -> list[str]:
    if count <= 0:
        return []
    ranked = [
        (statistics.median(samples[-_MAX_SAMPLES:]), node_id)
        for node_id, samples in history.items()
        if len(samples) >= _MIN_SAMPLES
    ]
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [node_id for _median, node_id in ranked[:count]]


def merge_previous_samples(repo_root: Path) -> dict[str, list[float]]:
    """Fold the previous run's JSONL into five-sample history."""
    history_path = repo_root / HISTORY_REL
    samples_path = repo_root / SAMPLES_REL
    history = load_history(history_path)
    latest: dict[str, float] = {}
    if samples_path.is_file():
        for raw_line in samples_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(raw_line)
                node_id = str(event.get("node_id") or "")
                duration = float(event.get("duration_sec"))
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if node_id and duration >= 0:
                latest[node_id] = duration
    for node_id, duration in latest.items():
        history[node_id] = [*history.get(node_id, []), duration][-_MAX_SAMPLES:]
    if latest:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = history_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(history_path)
    samples_path.unlink(missing_ok=True)
    return history


def spread_slow_indices(
    collection: list[str],
    ranked_nodeids: list[str],
    worker_count: int,
) -> list[list[int]]:
    """Assign ranked slow tests round-robin, one per worker where possible."""
    assignments = [[] for _ in range(max(0, worker_count))]
    if not assignments:
        return assignments
    index_by_node = {node_id: index for index, node_id in enumerate(collection)}
    indices = [index_by_node[node_id] for node_id in ranked_nodeids if node_id in index_by_node]
    for position, index in enumerate(indices):
        assignments[position % worker_count].append(index)
    return assignments


class SlowFirstWorkStealingScheduling(WorkStealingScheduling):
    def __init__(self, config: Any, log: Any, ranked_nodeids: list[str]) -> None:
        super().__init__(config, log)
        self._ranked_nodeids = ranked_nodeids

    def schedule(self) -> None:
        if self.collection is not None:
            self.check_schedule()
            return
        if not self._check_nodes_have_same_collection():
            self.log("**Different tests collected, aborting run**")
            return

        self.collection = next(iter(self.node2collection.values()))
        if not self.collection:
            return
        nodes = [node for node in self.node2pending if not node.shutting_down]
        assignments = spread_slow_indices(self.collection, self._ranked_nodeids, len(nodes))
        slow_indices = {index for assignment in assignments for index in assignment}
        self.pending[:] = [index for index in range(len(self.collection)) if index not in slow_indices]

        for node, indices in zip(nodes, assignments):
            if not indices:
                continue
            self.node2pending[node].extend(indices)
            node.send_runtest_some(indices)
        if slow_indices:
            logging.getLogger("nightly").info(
                "Nightly slow-first: distributed %d tests across %d workers",
                len(slow_indices),
                min(len(slow_indices), len(nodes)),
            )
        self.check_schedule()


def pytest_configure(config: Any) -> None:
    global _sample_path, _write_samples
    raw_path = (os.environ.get(_HISTORY_ENV) or "").strip()
    history_path = Path(raw_path) if raw_path else None
    _sample_path = history_path.parent / Path(SAMPLES_REL).name if history_path else None
    is_worker = isinstance(getattr(config, "workerinput", None), dict)
    num_processes = getattr(config.option, "numprocesses", None)
    _write_samples = _sample_path is not None and bool(num_processes) and not is_worker


def pytest_xdist_make_scheduler(config: Any, log: Any) -> SlowFirstWorkStealingScheduling:
    raw_path = (os.environ.get(_HISTORY_ENV) or "").strip()
    history = load_history(Path(raw_path)) if raw_path else {}
    ranked = ranked_slow_nodeids(history, resolve_slow_first_count())
    return SlowFirstWorkStealingScheduling(config, log, ranked)


def pytest_runtest_logreport(report: Any) -> None:
    if not _write_samples or _sample_path is None or getattr(report, "when", "") != "call":
        return
    node_id = str(getattr(report, "nodeid", "") or "")
    if not node_id:
        return
    event = {
        "node_id": node_id,
        "duration_sec": float(getattr(report, "duration", 0.0) or 0.0),
    }
    _sample_path.parent.mkdir(parents=True, exist_ok=True)
    with _sample_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


SLOW_HISTORY_ENV: Final = _HISTORY_ENV

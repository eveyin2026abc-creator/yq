"""Persist pytest failures immediately so process termination cannot erase evidence."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

_JOURNAL_ENV: Final = "MSMODELING_NIGHTLY_FAILURE_JOURNAL"
_journal_path: Path | None = None
_worker_id = "master"
_write_enabled = False


def pytest_configure(config: Any) -> None:
    """Enable one atomic JSONL writer per serial process or xdist worker."""
    global _journal_path, _worker_id, _write_enabled

    raw_path = (os.environ.get(_JOURNAL_ENV) or "").strip()
    _journal_path = Path(raw_path) if raw_path else None
    worker_input = getattr(config, "workerinput", None)
    if isinstance(worker_input, dict):
        _worker_id = str(worker_input.get("workerid", "worker"))
        _write_enabled = _journal_path is not None
        return

    num_processes = getattr(config.option, "numprocesses", None)
    _worker_id = "master"
    _write_enabled = _journal_path is not None and not num_processes


def _append_failure(report: Any, *, phase: str) -> None:
    if not _write_enabled or _journal_path is None or not getattr(report, "failed", False):
        return

    traceback = getattr(report, "longreprtext", "") or str(getattr(report, "longrepr", ""))
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "worker": _worker_id,
        "node_id": str(getattr(report, "nodeid", "")),
        "phase": phase,
        "outcome": str(getattr(report, "outcome", "failed")),
        "duration_sec": float(getattr(report, "duration", 0.0)),
        "traceback": traceback,
    }
    encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
    _journal_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(_journal_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(fd, remaining)
            remaining = remaining[written:]
        os.fsync(fd)
    finally:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def pytest_runtest_logreport(report: Any) -> None:
    """Persist setup/call/teardown failures as soon as pytest reports them."""
    _append_failure(report, phase=str(getattr(report, "when", "call")))


def pytest_collectreport(report: Any) -> None:
    """Persist collection errors, which never reach runtest hooks."""
    _append_failure(report, phase="collect")

#!/usr/bin/env python3
"""Local calendar for msmodeling nightly logs."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
TRACKING_PATH = Path(os.environ.get("NIGHTLY_BOARD_TRACKING", ROOT / "tracking.json"))
LOG_DIR = Path(os.environ.get("NIGHTLY_BOARD_LOG_DIR", "/tmp"))
LOG_NAME = re.compile(r"^nightly-(?P<label>.+)-(?P<ymd>\d{8})-(?P<hhmm>\d{4})\.log$")
EXIT_RE = re.compile(
    r"Nightly exit=(\d+): passed=(\d+) failed=(\d+) errors=(\d+) duration=(\d+)s"
)
COV_RE = re.compile(r"Coverage gate (passed|failed): line=([0-9.]+)% branch=([0-9.]+)%")
FAILED_RE = re.compile(r"Failed: (.+)$")
BEIJING = timezone(timedelta(hours=8))
HOST = os.environ.get("NIGHTLY_BOARD_HOST", "0.0.0.0")
PORT = int(os.environ.get("NIGHTLY_BOARD_PORT", "8766"))
ISSUE_CACHE: dict[int, tuple[float, dict]] = {}
REPO = "Ascend/msmodeling"


def beijing_today() -> str:
    return datetime.now(BEIJING).date().isoformat()


def load_tracking() -> dict:
    if not TRACKING_PATH.is_file():
        return {"runs": {}}
    payload = json.loads(TRACKING_PATH.read_text(encoding="utf-8"))
    payload.setdefault("runs", {})
    return payload


def refine_error(text: str, node_id: str, fallback: str) -> str:
    lines = text.splitlines()
    marker = f"FAILED {node_id} - "
    for index, line in enumerate(lines):
        if marker not in line:
            continue
        window = lines[index : index + 80]
        for candidate in window:
            if "ValueError:" in candidate:
                snippet = candidate.split("ValueError:", 1)[1].strip()
                snippet = snippet.replace("[non-benchmark]", "").strip()
                if len(snippet) > 500:
                    snippet = snippet[:500] + "…"
                return f"ValueError: {snippet}"
        matched = line.split(marker, 1)[1].strip()
        if len(matched) > 500:
            matched = matched[:500] + "…"
        return matched
    return fallback


def parse_failed(line: str) -> dict:
    body = line.split("Failed: ", 1)[1].strip()
    parts = [part.strip() for part in body.split(" | ")]
    fields = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value.strip()
    error = fields.get("error", "")
    if len(error) > 500:
        error = error[:500] + "…"
    return {
        "node_id": parts[0],
        "status": fields.get("status", ""),
        "commit": fields.get("commit", ""),
        "author": fields.get("author", ""),
        "subject": fields.get("subject", ""),
        "error": error,
    }


def parse_log(path: Path) -> dict | None:
    match = LOG_NAME.match(path.name)
    if match is None:
        return None
    label = match.group("label")
    branch = "26.2.0" if label.startswith("26.2.0") else "master" if label.startswith("master") else label
    ymd = match.group("ymd")
    date = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
    text = path.read_text(encoding="utf-8", errors="replace")
    exit_match = None
    coverage = None
    failures = []
    for line in text.splitlines():
        found = EXIT_RE.search(line)
        if found:
            exit_match = found
        cov = COV_RE.search(line)
        if cov:
            coverage = {
                "gate": cov.group(1),
                "line": float(cov.group(2)),
                "branch": float(cov.group(3)),
            }
        if "Failed: " in line and "status=" in line:
            failure = parse_failed(line)
            failure["error"] = refine_error(text, failure["node_id"], failure["error"])
            failures.append(failure)
    if exit_match is None:
        return None
    exit_code = int(exit_match.group(1))
    return {
        "log": path.name,
        "label": label,
        "branch": branch,
        "date": date,
        "started": f"{date} {match.group('hhmm')[:2]}:{match.group('hhmm')[2:]}",
        "exit_code": exit_code,
        "passed": int(exit_match.group(2)),
        "failed": int(exit_match.group(3)),
        "errors": int(exit_match.group(4)),
        "duration_sec": int(exit_match.group(5)),
        "coverage": coverage,
        "failures": failures,
    }


def load_runs() -> list[dict]:
    tracking = load_tracking()["runs"]
    runs = []
    for path in sorted(LOG_DIR.glob("nightly-*.log")):
        run = parse_log(path)
        if run is None:
            continue
        extra = tracking.get(path.name) or {}
        run["commit"] = extra.get("commit") or ""
        run["ci_miss_reason"] = extra.get("ci_miss_reason") or ""
        run["issue_numbers"] = [int(item) for item in extra.get("issues") or []]
        runs.append(run)
    return runs


def issue_state(number: int) -> dict:
    now = datetime.now(timezone.utc).timestamp()
    cached = ISSUE_CACHE.get(number)
    if cached and now - cached[0] < 60:
        return cached[1]
    try:
        completed = subprocess.run(
            ["gitcode", "issue", "view", str(number), "-R", REPO, "--json"],
            check=False,
            text=True,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        payload = {
            "number": number,
            "state": "unknown",
            "closed": None,
            "title": "",
            "html_url": f"https://gitcode.com/{REPO}/issues/{number}",
            "assignees": [],
            "error": str(exc),
        }
        return payload
    if completed.returncode != 0:
        payload = {
            "number": number,
            "state": "unknown",
            "closed": None,
            "title": "",
            "html_url": f"https://gitcode.com/{REPO}/issues/{number}",
            "assignees": [],
            "error": (completed.stderr or completed.stdout or "gitcode issue view failed")[:300],
        }
        return payload
    data = json.loads(completed.stdout)
    state = data.get("state") or "unknown"
    payload = {
        "number": number,
        "state": state,
        "closed": state == "closed",
        "title": data.get("title") or "",
        "html_url": data.get("html_url") or f"https://gitcode.com/{REPO}/issues/{number}",
        "assignees": [item.get("login") for item in data.get("assignees") or [] if item.get("login")],
        "error": "",
    }
    ISSUE_CACHE[number] = (now, payload)
    return payload


def attach_issues(run: dict) -> dict:
    issues = [issue_state(number) for number in run["issue_numbers"]]
    if not issues:
        loop = "未建 issue"
    elif all(item["closed"] is True for item in issues):
        loop = "已闭环"
    elif any(item["closed"] is False for item in issues):
        loop = "未闭环"
    else:
        loop = "状态未知"
    run = dict(run)
    run["issues"] = issues
    run["loop"] = loop
    return run


def day_status(runs: list[dict]) -> str:
    master = [run for run in runs if run["branch"] == "master" and run["label"] == "master"]
    chosen = master or [run for run in runs if run["branch"] == "master"] or runs
    latest = max(chosen, key=lambda run: run["started"])
    code = latest["exit_code"]
    if code == 0 and latest["failed"] == 0 and latest["errors"] == 0:
        return "pass"
    if code == 124:
        return "timeout"
    return "fail"


def month_payload(year: int, month: int) -> dict:
    runs = load_runs()
    prefix = f"{year:04d}-{month:02d}-"
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        if run["date"].startswith(prefix):
            grouped.setdefault(run["date"], []).append(run)
    days = {
        date: {
            "status": day_status(items),
            "runs": len(items),
            "failed": sum(item["failed"] for item in items),
        }
        for date, items in grouped.items()
    }
    return {"year": year, "month": month, "today": beijing_today(), "days": days}


def day_payload(date: str) -> dict:
    runs = [attach_issues(run) for run in load_runs() if run["date"] == date]
    runs.sort(key=lambda run: (run["branch"] != "master", run["label"] != "master", run["started"]))
    return {"date": date, "today": beijing_today(), "runs": runs}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            page = (ROOT / "index.html").read_bytes()
            self._send(200, page, "text/html; charset=utf-8")
            return
        if parsed.path == "/config.js":
            config = (ROOT / "config.js").read_bytes()
            self._send(200, config, "application/javascript; charset=utf-8")
            return
        if parsed.path == "/api/month":
            today = datetime.now(BEIJING).date()
            year = int(query.get("year", [str(today.year)])[0])
            month = int(query.get("month", [str(today.month)])[0])
            body = json.dumps(month_payload(year, month), ensure_ascii=False).encode()
            self._send(200, body, "application/json; charset=utf-8")
            return
        if parsed.path == "/api/day":
            date = query.get("date", [beijing_today()])[0]
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                self._send(400, b'{"error":"bad date"}', "application/json")
                return
            body = json.dumps(day_payload(date), ensure_ascii=False).encode()
            self._send(200, body, "application/json; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"nightly board http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

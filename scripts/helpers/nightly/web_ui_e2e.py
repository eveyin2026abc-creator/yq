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

"""Run the web console Playwright suite after the pytest waves.

The suite is not a pytest node. Nightly invokes ``npm run test`` in
``tests/e2e/web_ui`` when that directory contains ``*.e2e.ts`` specs, then
parses the JSON report. No specs means a quiet skip (no install, no servers).
Install steps sit outside the test budget. Failures stay out of pytest
attribution.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scripts.helpers._paths import REPO_ROOT

# Playwright test only. npm install and the Chromium download are not included.
WEB_UI_E2E_TEST_TIMEOUT_SECONDS = 900
_INSTALL_TIMEOUT_SECONDS = 900
_BROWSER_INSTALL_TIMEOUT_SECONDS = 600
_LOG_TAIL_CHARS = 4000
_SPEC_PREFIX = "tests/e2e/web_ui/"


@dataclass(frozen=True, slots=True)
class WebUiE2EResult:
    """Outcome of one web UI e2e attempt."""

    exit_code: int
    passed: int
    failed: int
    ran: bool
    note: str
    failed_specs: tuple[str, ...] = ()


def parse_playwright_report(payload: dict[str, object]) -> tuple[int, int, tuple[str, ...]]:
    """Return passed, failed, and failed spec ids from a Playwright JSON report."""
    passed = 0
    failed = 0
    failed_specs: list[str] = []

    def walk(suite: dict[str, object], file_name: str, titles: tuple[str, ...]) -> None:
        nonlocal passed, failed
        current_file = str(suite.get("file") or file_name)
        title = str(suite.get("title") or "")
        next_titles = titles
        if title and title != current_file:
            next_titles = (*titles, title)
        for spec in suite.get("specs") or []:
            if not isinstance(spec, dict):
                continue
            spec_title = str(spec.get("title") or "")
            full_title = " › ".join((*next_titles, spec_title)).strip(" ›")
            ok = spec.get("ok")
            if ok is None:
                ok = _last_result_ok(spec)
            if ok:
                passed += 1
            else:
                failed += 1
                rel = current_file.removeprefix("./")
                failed_specs.append(f"{_SPEC_PREFIX}{rel}::{full_title}")
        for child in suite.get("suites") or []:
            if isinstance(child, dict):
                walk(child, current_file, next_titles)

    for suite in payload.get("suites") or []:
        if isinstance(suite, dict):
            walk(suite, "", ())
    return passed, failed, tuple(failed_specs)


def run_web_ui_e2e(
    logger: logging.Logger,
    *,
    skip: bool,
    repo_root: Path | None = None,
) -> WebUiE2EResult:
    """Install the stack if needed, then run the Playwright suite."""
    if skip:
        note = "Web UI e2e skipped: pytest timed out"
        logger.info(note)
        return WebUiE2EResult(exit_code=0, passed=0, failed=0, ran=False, note=note)

    root = repo_root or REPO_ROOT
    e2e_dir = root / "tests" / "e2e" / "web_ui"
    frontend_dir = root / "web_ui" / "frontend"
    if not (e2e_dir / "package.json").is_file():
        note = "Web UI e2e infra: tests/e2e/web_ui/package.json is missing"
        logger.error(note)
        return WebUiE2EResult(exit_code=1, passed=0, failed=0, ran=False, note=note)
    if not any(e2e_dir.rglob("*.e2e.ts")):
        logger.info("Web UI e2e skipped: no specs")
        return WebUiE2EResult(exit_code=0, passed=0, failed=0, ran=False, note="")
    if shutil.which("npm") is None or shutil.which("node") is None:
        note = "Web UI e2e infra: node and npm are required"
        logger.error(note)
        return WebUiE2EResult(exit_code=1, passed=0, failed=0, ran=False, note=note)

    frontend_install = ["npm", "ci"] if (frontend_dir / "package-lock.json").is_file() else ["npm", "install"]
    prepare_steps = (
        (frontend_install, frontend_dir, _INSTALL_TIMEOUT_SECONDS, "frontend npm install"),
        (["npm", "install"], e2e_dir, _INSTALL_TIMEOUT_SECONDS, "e2e npm install"),
        (
            ["npx", "playwright", "install", "chromium"],
            e2e_dir,
            _BROWSER_INSTALL_TIMEOUT_SECONDS,
            "playwright chromium",
        ),
    )
    for command, cwd, timeout, label in prepare_steps:
        code, output = _run_command(command, cwd=cwd, timeout=timeout, env=os.environ.copy())
        if code != 0:
            note = f"Web UI e2e infra: {label} failed ({code})"
            logger.error("%s\n%s", note, output[-_LOG_TAIL_CHARS:])
            return WebUiE2EResult(exit_code=1, passed=0, failed=0, ran=False, note=note)

    report_path = e2e_dir / "e2e-results.json"
    report_path.unlink(missing_ok=True)
    test_env = os.environ.copy()
    test_env["CI"] = "1"
    test_env["MSMODELING_E2E_JSON"] = str(report_path)
    _prepend_user_chromium_libs(test_env)
    logger.info("Web UI e2e: npm run test (budget %ss)", WEB_UI_E2E_TEST_TIMEOUT_SECONDS)
    code, output = _run_command(
        ["npm", "run", "test"],
        cwd=e2e_dir,
        timeout=WEB_UI_E2E_TEST_TIMEOUT_SECONDS,
        env=test_env,
    )
    if output:
        logger.info("Web UI e2e output tail:\n%s", output[-_LOG_TAIL_CHARS:])
    if code == 124:
        note = "Web UI e2e timed out"
        logger.error(note)
        return WebUiE2EResult(exit_code=1, passed=0, failed=0, ran=True, note=note)

    passed, failed, failed_specs = _read_report(report_path, logger)
    if code != 0 and failed == 0 and passed == 0:
        note = f"Web UI e2e infra: npm run test exited {code}"
        return WebUiE2EResult(exit_code=1, passed=0, failed=0, ran=False, note=note)
    note = "" if code == 0 else f"Web UI e2e failed: {failed}"
    if failed_specs:
        logger.error("%s", note)
        for spec in failed_specs:
            logger.error("Web UI e2e failed spec: %s", spec)
    else:
        logger.info("Web UI e2e passed=%d failed=%d", passed, failed)
    return WebUiE2EResult(
        exit_code=0 if code == 0 else 1,
        passed=passed,
        failed=failed,
        ran=True,
        note=note,
        failed_specs=failed_specs,
    )


def _last_result_ok(spec: dict[str, object]) -> bool:
    tests = spec.get("tests")
    if not isinstance(tests, list) or not tests:
        return False
    last = tests[-1]
    if not isinstance(last, dict):
        return False
    results = last.get("results")
    if not isinstance(results, list) or not results:
        return False
    final = results[-1]
    if not isinstance(final, dict):
        return False
    return final.get("status") in {"passed", "skipped"}


def _prepend_user_chromium_libs(env: dict[str, str]) -> None:
    """Use a user-prefix libasound when the system package is not installed.

    Headless Chromium refuses to start without ``libasound.so.2``. Hosts that
    cannot install the deb system-wide can extract it under
    ``~/.local/playwright-libs/usr/lib/<triplet>/``.
    """
    lib_root = Path.home() / ".local" / "playwright-libs" / "usr" / "lib"
    extra = _library_dirs_containing(lib_root, "libasound.so.2")
    if not extra:
        return
    current = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(extra + ([current] if current else []))


def _library_dirs_containing(lib_root: Path, soname: str) -> list[str]:
    if not lib_root.is_dir():
        return []
    return [
        str(path)
        for path in sorted(lib_root.iterdir())
        if path.is_dir() and (path / soname).is_file()
    ]


def _read_report(
    report_path: Path,
    logger: logging.Logger,
) -> tuple[int, int, tuple[str, ...]]:
    if not report_path.is_file():
        logger.error("Web UI e2e JSON report missing: %s", report_path)
        return 0, 0, ()
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.exception("Web UI e2e JSON report is not valid JSON")
        return 0, 0, ()
    if not isinstance(payload, dict):
        return 0, 0, ()
    return parse_playwright_report(payload)


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    env: dict[str, str],
) -> tuple[int, str]:
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        return 124, ""
    return proc.returncode or 0, output or ""


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait(timeout=10)

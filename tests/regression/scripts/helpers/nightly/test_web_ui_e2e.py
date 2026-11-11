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

"""Web UI e2e stage: report parsing and how it changes the nightly exit."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.helpers._config import Config
from scripts.helpers.nightly.feishu_notifier import build_feishu_card_payload
from scripts.helpers.nightly.main import _apply_web_ui_e2e_exit, _run_nightly_pipeline
from scripts.helpers.nightly.report_models import FeishuReportInput
from scripts.helpers.nightly.web_ui_e2e import WebUiE2EResult, parse_playwright_report, run_web_ui_e2e


def test_parse_playwright_report_counts_and_ids() -> None:
    passed, failed, specs = parse_playwright_report(
        {
            "suites": [
                {
                    "title": "console-text-generate.e2e.ts",
                    "file": "console-text-generate.e2e.ts",
                    "suites": [
                        {
                            "title": "Console · text_generate lifecycle",
                            "specs": [
                                {"title": "num-devices=9", "ok": True, "tests": []},
                                {"title": "sample failed", "ok": False, "tests": []},
                            ],
                        }
                    ],
                }
            ]
        }
    )
    assert passed == 1
    assert failed == 1
    assert specs == (
        "tests/e2e/web_ui/console-text-generate.e2e.ts::Console · text_generate lifecycle › sample failed",
    )


def test_e2e_failure_does_not_override_timeout_or_attribution() -> None:
    failed = WebUiE2EResult(exit_code=1, passed=5, failed=1, ran=True, note="Web UI e2e failed: 1")
    assert _apply_web_ui_e2e_exit(124, failed, timed_out=True, status_note="timed out") == (124, "timed out")
    assert _apply_web_ui_e2e_exit(3, failed, timed_out=False, status_note="") == (3, "")
    code, note = _apply_web_ui_e2e_exit(0, failed, timed_out=False, status_note="")
    assert code == 1
    assert note == "Web UI e2e failed: 1"


def test_missing_suite_is_infra_and_does_not_run_npm(tmp_path: Path) -> None:
    result = run_web_ui_e2e(logging.getLogger("nightly-test"), skip=False, repo_root=tmp_path)
    assert result.ran is False
    assert result.exit_code == 1
    assert "package.json is missing" in result.note


def test_skip_does_not_touch_the_tree(tmp_path: Path) -> None:
    result = run_web_ui_e2e(logging.getLogger("nightly-test"), skip=True, repo_root=tmp_path)
    assert result == WebUiE2EResult(
        exit_code=0,
        passed=0,
        failed=0,
        ran=False,
        note="Web UI e2e skipped: pytest timed out",
    )


def test_pipeline_e2e_failure_keeps_pytest_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.helpers.nightly import main as nightly_main

    monkeypatch.setattr(nightly_main, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(nightly_main, "_run_pytest_waves", lambda *a, **k: (0, "===== 2 passed in 1.00s =====", 0, ""))
    monkeypatch.setattr(nightly_main, "_coverage_summary", lambda _cfg: None)
    monkeypatch.setattr(nightly_main, "_run_config_drift_check", lambda: ())
    monkeypatch.setattr(
        nightly_main,
        "run_web_ui_e2e",
        lambda *_a, **_k: WebUiE2EResult(
            exit_code=1,
            passed=5,
            failed=1,
            ran=True,
            note="Web UI e2e failed: 1",
            failed_specs=("tests/e2e/web_ui/console-video-generate.e2e.ts::sample-step=4",),
        ),
    )
    reports: list[FeishuReportInput] = []
    monkeypatch.setattr(nightly_main, "push_feishu_report", lambda _url, report: reports.append(report))
    cfg = Config(
        base_branch="master",
        line_threshold=80.0,
        branch_threshold=60.0,
        feishu_webhook_url="https://example.com/hook",
    )
    exit_code = _run_nightly_pipeline(
        logging.getLogger("nightly-test"),
        cfg,
        "https://example.com/hook",
        "",
        timeout_seconds=3000,
    )
    assert exit_code == 1
    assert reports[0].failed == 0
    assert reports[0].e2e_passed == 5
    assert reports[0].e2e_failed == 1
    assert reports[0].status_note == "Web UI e2e failed: 1"
    card = build_feishu_card_payload(reports[0])
    text = str(card)
    assert "Web UI e2e" in text
    assert "console-video-generate.e2e.ts" in text

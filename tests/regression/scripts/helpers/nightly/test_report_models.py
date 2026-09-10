"""Tests for nightly.report_models — CoverageSummary, EnvInfo, FailureBlame."""

from __future__ import annotations

from scripts.helpers.nightly.report_models import (
    AttributionConclusion,
    CoverageSummary,
    EnvInfo,
    FailureBlame,
    FeishuReportInput,
    NotReproducedCause,
    classify_not_reproduced_cause,
)


def test_env_info_fields() -> None:
    e = EnvInfo(commit="abc123", branch="main", timestamp="2026-01-01T00:00:00Z")
    assert e.commit == "abc123"
    assert e.branch == "main"
    assert e.timestamp == "2026-01-01T00:00:00Z"


def test_coverage_summary() -> None:
    c = CoverageSummary(
        line_percent=85.5,
        branch_percent=72.3,
        line_threshold=70.0,
        branch_threshold=50.0,
        gate_passed=True,
        message="passed",
    )
    assert c.line_percent == 85.5
    assert c.gate_passed is True


def test_failure_blame_fields_and_conclusion_helpers() -> None:
    assert AttributionConclusion.FIRST_BAD == "first_bad"
    assert isinstance(AttributionConclusion.FIRST_BAD, str)
    blame = FailureBlame(
        node_id="tests/a.py::test_x",
        commit_id="abc1234",
        author="alice",
        subject="add test",
        conclusion=AttributionConclusion.FIRST_BAD,
        last_reason="AssertionError: x",
    )
    assert blame.node_id == "tests/a.py::test_x"
    assert blame.attributed is True
    assert blame.needs_human is False

    need_human = FailureBlame(
        node_id="tests/b.py::test_y",
        commit_id="unknown",
        author="unknown",
        subject="Still failing after 7-day lookback; needs human follow-up",
        conclusion=AttributionConclusion.NEED_HUMAN,
    )
    assert need_human.needs_human is True
    assert need_human.attributed is False


def test_feishu_report_input_accepts_failure_blames() -> None:
    report = FeishuReportInput(
        timestamp="2026-01-01T00:00:00Z",
        branch="main",
        commit="abc",
        passed=0,
        failed=1,
        errors=0,
        duration_sec=1.0,
        overall_exit=1,
        coverage_line_percent=None,
        coverage_branch_percent=None,
        coverage_line_threshold=None,
        coverage_branch_threshold=None,
        coverage_gate_passed=None,
        failure_blames=(
            FailureBlame(
                node_id="tests/a.py::test_x",
                commit_id="deadbeef",
                author="bob",
                subject="fix test",
                conclusion=AttributionConclusion.FIRST_BAD,
            ),
        ),
        pipeline_log_url="https://ci.example/log/1",
        timed_out=False,
        status_note="",
    )
    assert report.failure_blames[0].commit_id == "deadbeef"
    assert report.pipeline_log_url.startswith("https://")


def test_classify_not_reproduced_cause_network_timeout_unknown() -> None:
    assert (
        classify_not_reproduced_cause(
            "OSError: We couldn't connect to 'https://hf-mirror.com' to load the files"
        )
        == NotReproducedCause.NETWORK
    )
    assert (
        classify_not_reproduced_cause("httpx.HTTPStatusError: Client error '429 Too Many Requests'")
        == NotReproducedCause.NETWORK
    )
    assert classify_not_reproduced_cause("TimeoutError: pytest timeout") == NotReproducedCause.TIMEOUT
    assert classify_not_reproduced_cause("timed out after 7200s; exit=124") == NotReproducedCause.TIMEOUT
    assert (
        classify_not_reproduced_cause("assert (708331.47 - 708324.28) < 5") == NotReproducedCause.UNKNOWN
    )
    assert (
        classify_not_reproduced_cause("TypeError: not all arguments converted during string formatting")
        == NotReproducedCause.UNKNOWN
    )
    assert classify_not_reproduced_cause("") == NotReproducedCause.UNKNOWN
    assert (
        classify_not_reproduced_cause("429 Too Many Requests; retry timed out") == NotReproducedCause.NETWORK
    )

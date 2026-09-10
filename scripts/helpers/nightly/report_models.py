"""Report domain models for nightly pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Final


class AttributionConclusion(str, Enum):
    """Per-failure conclusion for Feishu + exit-code policy."""

    FIRST_BAD = "first_bad"
    NEED_HUMAN = "need_human"
    CANNOT_REPRODUCE = "cannot_reproduce"
    UNCOLLECTIBLE = "uncollectible"


class NotReproducedCause(str, Enum):
    """Why a primary failure did not reproduce at HEAD. Keep labels traceback-literal."""

    NETWORK = "network"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


_NETWORK_CAUSE_RE: Final = re.compile(
    r"429|too many requests|couldn't connect|could not connect|"
    r"hfhubhttperror|localentrynotfounderror|"
    r"connection(?:error|refused|reset|aborted)|"
    r"nameresolutionerror|sslerror|remotedisconnected|"
    r"max retries exceeded|proxyerror|"
    r"failed to establish|temporary failure in name resolution|"
    r"huggingface\.co|hf-mirror",
    re.IGNORECASE,
)
_TIMEOUT_CAUSE_RE: Final = re.compile(
    r"\btimeout(?:error|expired)?\b|timed out|pytest[- ]timeout|"
    r"cancelled by timeout|deadline exceeded|\bexit(?:\s*code)?\s*[=:]?\s*124\b",
    re.IGNORECASE,
)


def classify_not_reproduced_cause(error_text: str) -> NotReproducedCause:
    """Classify a Not-reproduced failure from the primary-wave error text.

    Network wins when both network and timeout tokens appear (Hub retries often
    mention both). Timing assertions such as ``elapsed < 5`` stay unknown.
    """
    text = error_text.strip()
    if not text:
        return NotReproducedCause.UNKNOWN
    if _NETWORK_CAUSE_RE.search(text):
        return NotReproducedCause.NETWORK
    if _TIMEOUT_CAUSE_RE.search(text):
        return NotReproducedCause.TIMEOUT
    return NotReproducedCause.UNKNOWN


@dataclass(frozen=True, slots=True)
class EnvInfo:
    commit: str
    branch: str
    timestamp: str


@dataclass(frozen=True, slots=True)
class MapCoverageSummary:
    test_nodes: int
    symbol_refs: int


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    line_percent: float
    branch_percent: float
    line_threshold: float
    branch_threshold: float
    gate_passed: bool
    message: str


@dataclass(frozen=True, slots=True)
class FailureBlame:
    node_id: str
    commit_id: str
    author: str
    subject: str
    conclusion: AttributionConclusion
    last_reason: str = ""
    cause: str = ""

    @property
    def attributed(self) -> bool:
        return self.conclusion == AttributionConclusion.FIRST_BAD

    @property
    def needs_human(self) -> bool:
        return self.conclusion in {
            AttributionConclusion.NEED_HUMAN,
            AttributionConclusion.UNCOLLECTIBLE,
        }


@dataclass(frozen=True, slots=True)
class FeishuReportInput:
    timestamp: str
    branch: str
    commit: str
    passed: int
    failed: int
    errors: int
    duration_sec: float
    overall_exit: int
    coverage_line_percent: float | None
    coverage_branch_percent: float | None
    coverage_line_threshold: float | None
    coverage_branch_threshold: float | None
    coverage_gate_passed: bool | None
    failure_blames: tuple[FailureBlame, ...] = ()
    drift_warnings: tuple[str, ...] = ()
    pipeline_log_url: str = ""
    infra_message: str = ""
    timed_out: bool = False
    status_note: str = ""
    observed_completed: int = 0
    summary_complete: bool = True

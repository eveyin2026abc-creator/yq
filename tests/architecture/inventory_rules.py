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

"""The layer and schedule rules used by the 9154-row register.

Rules run in the order recorded in ``docs/design/ut_refactor_implementation_plan.md``.
A later rule does not override an earlier one. ``待人工确认`` is not L2 or L3.
"""

from __future__ import annotations

from dataclasses import dataclass

_PENDING_SIGNATURE = "UserInputConfig build fields plus compilation_config"


@dataclass(frozen=True)
class NodeRecord:
    path: str
    markers: tuple[str, ...] = ()
    has_model_runner: bool = False
    has_build_model: bool = False


@dataclass(frozen=True)
class Classification:
    target_layer: str
    migration: str
    schedule: str
    evidence: str
    assertion_overlap: str
    build_signature: str


def schedule_for(record: NodeRecord) -> str:
    """Where the node runs today. The path rule wins over markers."""

    if record.path.startswith("tests/benchmark/"):
        return "benchmark"
    if "npu" in record.markers:
        return "excluded-npu"
    if "network" in record.markers:
        return "nightly-network"
    if "nightly" in record.markers:
        return "nightly"
    return "incremental"


def classify(record: NodeRecord) -> Classification:
    """Apply the seven layer rules in order."""

    schedule = schedule_for(record)
    if record.path.startswith("tests/smoke/"):
        return _classified("smoke", "保持", schedule, "path under tests/smoke/")
    if record.path.startswith("tests/benchmark/models/"):
        return _classified(
            "L1",
            "保持",
            schedule,
            "path under tests/benchmark/models/",
            assertion_overlap="numeric",
        )
    if record.path.startswith("tests/benchmark/"):
        return _classified("benchmark-op", "保持", schedule, "other path under tests/benchmark/")
    if "model_diagnostics/e2e/" in record.path:
        return _classified(
            "L1",
            "待归并到 L1",
            schedule,
            "path contains model_diagnostics/e2e/",
            assertion_overlap="diagnostics",
        )
    if record.path.startswith("tests/regression/scripts/") or record.path.startswith("tests/helpers/tests/"):
        return _classified("L3", "保持", schedule, "toolchain or helper contract path")
    if record.has_model_runner or record.has_build_model:
        evidence = "file text contains ModelRunner" if record.has_model_runner else "file text contains build_model"
        if record.has_model_runner and record.has_build_model:
            evidence = "file text contains ModelRunner and build_model"
        return _classified("L1", "待归并到 L1", schedule, evidence, assertion_overlap="未判定")
    return _classified("待人工确认", "待人工确认", schedule, "no rule matched")


def _classified(
    target_layer: str,
    migration: str,
    schedule: str,
    evidence: str,
    assertion_overlap: str = "",
) -> Classification:
    signature = _PENDING_SIGNATURE if target_layer == "L1" else ""
    return Classification(target_layer, migration, schedule, evidence, assertion_overlap, signature)

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

"""Synthetic L2 artifacts. None of these builders accept a model identity."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction

L2_KINDS = (
    "quant",
    "parallel",
    "fx",
    "runtime_event",
    "diagnostics",
    "replay",
    "microbench",
)


def _reject_identity(rejected: dict[str, object]) -> None:
    if rejected:
        names = ", ".join(sorted(rejected))
        raise TypeError(f"L2 builders do not accept a model identity ({names})")


@dataclass(frozen=True)
class QuantArtifact:
    kind: str
    linear_action: str
    attention_action: str


@dataclass(frozen=True)
class ParallelArtifact:
    kind: str
    world_size: int
    tensor_parallel: int
    expert_parallel: int


@dataclass(frozen=True)
class FxArtifact:
    kind: str
    node_ops: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeEventArtifact:
    kind: str
    op_name: str
    duration_us: int


@dataclass(frozen=True)
class DiagnosticsArtifact:
    kind: str
    region_id: str
    status: str


@dataclass(frozen=True)
class ReplayArtifact:
    kind: str
    num_queries: int
    query_len: int
    decode: bool


@dataclass(frozen=True)
class MicrobenchArtifact:
    kind: str
    op_name: str
    shape: tuple[int, ...]
    dtype: str


def quant_artifact(
    *,
    linear_action: QuantizeLinearAction,
    attention_action: QuantizeAttentionAction = QuantizeAttentionAction.DISABLED,
    **rejected: object,
) -> QuantArtifact:
    """One quant configuration with no weights and no model."""
    _reject_identity(rejected)
    return QuantArtifact("quant", linear_action.value, attention_action.value)


def parallel_artifact(
    *,
    world_size: int,
    tensor_parallel: int,
    expert_parallel: int = 1,
    **rejected: object,
) -> ParallelArtifact:
    """One parallel degree tuple. world_size must divide evenly."""
    _reject_identity(rejected)
    if min(world_size, tensor_parallel, expert_parallel) < 1:
        raise ValueError("parallel degrees must be positive")
    if world_size % tensor_parallel != 0 or world_size % expert_parallel != 0:
        raise ValueError("world_size must be divisible by the parallel degrees")
    return ParallelArtifact("parallel", world_size, tensor_parallel, expert_parallel)


def fx_artifact(**rejected: object) -> FxArtifact:
    """A one-node graph that does not trace a module."""
    _reject_identity(rejected)
    graph = torch.fx.Graph()
    value = graph.placeholder("x")
    graph.output(value)
    graph.lint()
    return FxArtifact("fx", tuple(node.op for node in graph.nodes))


def runtime_event_artifact(*, op_name: str, duration_us: int, **rejected: object) -> RuntimeEventArtifact:
    """One timed operator event with no runtime attached."""
    _reject_identity(rejected)
    if not op_name or duration_us < 0:
        raise ValueError("runtime event needs a name and a non-negative duration")
    return RuntimeEventArtifact("runtime_event", op_name, duration_us)


def diagnostics_artifact(*, region_id: str, status: str, **rejected: object) -> DiagnosticsArtifact:
    """One synthetic diagnostics finding."""
    _reject_identity(rejected)
    if status not in {"PASS", "FAIL"}:
        raise ValueError("diagnostics status must be PASS or FAIL")
    return DiagnosticsArtifact("diagnostics", region_id, status)


def replay_artifact(*, num_queries: int, query_len: int, decode: bool = False, **rejected: object) -> ReplayArtifact:
    """One replay request. It does not build or run a model."""
    _reject_identity(rejected)
    if num_queries < 1 or query_len < 1:
        raise ValueError("replay request needs a positive batch and query length")
    return ReplayArtifact("replay", num_queries, query_len, decode)


def microbench_artifact(*, op_name: str, shape: tuple[int, ...], dtype: str, **rejected: object) -> MicrobenchArtifact:
    """One microbench case described by op, shape, and dtype."""
    _reject_identity(rejected)
    if not op_name or not shape or any(dim < 1 for dim in shape):
        raise ValueError("microbench case needs an op name and a positive shape")
    return MicrobenchArtifact("microbench", op_name, shape, dtype)

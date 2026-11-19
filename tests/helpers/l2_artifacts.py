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
from typing import Callable

import torch

from tensor_cast.core.quantization.config import create_attention_quant_config, create_linear_quant_config
from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction
from tensor_cast.model_config import LinearQuantConfig, ParallelConfig

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
    linear_action: QuantizeLinearAction
    attention_action: QuantizeAttentionAction
    linear_config: LinearQuantConfig | None
    attention_config: object
    input_tensor: torch.Tensor
    weight: torch.Tensor


@dataclass(frozen=True)
class ParallelArtifact:
    kind: str
    config: ParallelConfig

    @property
    def world_size(self) -> int:
        return self.config.world_size

    @property
    def tensor_parallel(self) -> int:
        return self.config.tensor_parallel_size

    @property
    def expert_parallel(self) -> int:
        return self.config.expert_parallel_size


@dataclass(frozen=True)
class FxArtifact:
    kind: str
    node_ops: tuple[str, ...]
    graph_module: torch.fx.GraphModule


@dataclass(frozen=True)
class RuntimeEventArtifact:
    kind: str
    op_name: str
    duration_us: int
    op_invoke_info: RuntimeInvocation


@dataclass(frozen=True)
class _SyntheticOp:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class RuntimeInvocation:
    func: _SyntheticOp
    args: tuple[object, ...]
    kwargs: dict[str, object]
    out: object


@dataclass(frozen=True)
class DiagnosticsArtifact:
    kind: str
    region_id: str
    status: str
    finding: object


@dataclass(frozen=True)
class ReplayArtifact:
    kind: str
    num_queries: int
    query_len: int
    decode: bool
    row: dict[str, object]
    trace: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class MicrobenchArtifact:
    kind: str
    op_name: str
    shape: tuple[int, ...]
    dtype: str
    request: dict[str, object]


def quant_artifact(
    *,
    linear_action: QuantizeLinearAction,
    attention_action: QuantizeAttentionAction = QuantizeAttentionAction.DISABLED,
    input_shape: tuple[int, int] = (2, 4),
    output_features: int = 3,
    **rejected: object,
) -> QuantArtifact:
    """Real quant configs and the smallest tensors needed by a linear path."""
    _reject_identity(rejected)
    if len(input_shape) != 2 or min(input_shape) < 1 or output_features < 1:
        raise ValueError("quant input and output dimensions must be positive")
    return QuantArtifact(
        "quant",
        linear_action,
        attention_action,
        None if linear_action == QuantizeLinearAction.DISABLED else create_linear_quant_config(linear_action),
        None
        if attention_action == QuantizeAttentionAction.DISABLED
        else create_attention_quant_config(attention_action),
        torch.ones(input_shape, dtype=torch.float32),
        torch.ones((output_features, input_shape[-1]), dtype=torch.float32),
    )


def parallel_artifact(
    *,
    world_size: int,
    tensor_parallel: int,
    expert_parallel: int = 1,
    **rejected: object,
) -> ParallelArtifact:
    """A product ParallelConfig; invalid topologies fail in product validation."""
    _reject_identity(rejected)
    config = ParallelConfig(
        world_size=world_size,
        tensor_parallel_size=tensor_parallel,
        expert_parallel_size=expert_parallel,
    )
    return ParallelArtifact("parallel", config)


def fx_artifact(
    *,
    trace_func: Callable[..., object] | None = None,
    trace_args: tuple[object, ...] = (),
    **rejected: object,
) -> FxArtifact:
    """A synthetic FX graph, optionally traced from one component callable."""
    _reject_identity(rejected)
    if trace_func is not None:
        from torch.fx.experimental.proxy_tensor import make_fx

        graph_module = make_fx(trace_func)(*trace_args)
        return FxArtifact("fx", tuple(node.op for node in graph_module.graph.nodes), graph_module)
    graph = torch.fx.Graph()
    value = graph.placeholder("x")
    graph.output(value)
    graph.lint()
    graph_module = torch.fx.GraphModule({}, graph)
    return FxArtifact("fx", tuple(node.op for node in graph.nodes), graph_module)


def runtime_event_artifact(
    *,
    op_name: str,
    duration_us: int,
    args: tuple[object, ...] = (),
    kwargs: dict[str, object] | None = None,
    out: object = None,
    **rejected: object,
) -> RuntimeEventArtifact:
    """One runtime-compatible operator event without a model or runtime."""
    _reject_identity(rejected)
    if not op_name or duration_us < 0:
        raise ValueError("runtime event needs a name and a non-negative duration")
    invocation = RuntimeInvocation(
        func=_SyntheticOp(op_name),
        args=args,
        kwargs={} if kwargs is None else kwargs,
        out=out,
    )
    return RuntimeEventArtifact("runtime_event", op_name, duration_us, invocation)


def diagnostics_artifact(
    *,
    region_id: str,
    status: str,
    layer_index: int | None = 0,
    stage_id: str = "synthetic",
    rule_id: str = "synthetic.rule",
    comparison_kind: str = "contract",
    message: str = "synthetic diagnostic finding",
    expected: object = None,
    actual: object = None,
    left_evidence: tuple[object, ...] = (),
    right_evidence: tuple[object, ...] = (),
    **rejected: object,
) -> DiagnosticsArtifact:
    """One real diagnostics Finding built without a model identity."""
    _reject_identity(rejected)
    from tools.model_diagnostics.domain import Finding, FindingStatus

    try:
        finding_status = FindingStatus[status]
    except KeyError as error:
        raise ValueError(f"invalid diagnostics status: {status}") from error
    finding = Finding(
        region_id=region_id,
        layer_index=layer_index,
        stage_id=stage_id,
        rule_id=rule_id,
        comparison_kind=comparison_kind,
        status=finding_status,
        message_code=f"synthetic.{finding_status.value}",
        message=message,
        expected=expected,
        actual=actual,
        left_evidence=left_evidence,
        right_evidence=right_evidence,
    )
    return DiagnosticsArtifact("diagnostics", region_id, status, finding)


def replay_artifact(
    *,
    num_queries: int,
    query_len: int,
    decode: bool = False,
    fields: dict[str, object] | None = None,
    **rejected: object,
) -> ReplayArtifact:
    """One replay row plus its scheduling dimensions, without a model."""
    _reject_identity(rejected)
    if num_queries < 1 or query_len < 1:
        raise ValueError("replay request needs a positive batch and query length")
    row = {
        "num_queries": num_queries,
        "query_len": query_len,
        "decode": decode,
        **({} if fields is None else fields),
    }
    trace = (
        {
            "event": "request",
            "num_queries": num_queries,
            "query_len": query_len,
            "phase": "decode" if decode else "prefill",
        },
        {"event": "operator", "row": row},
    )
    return ReplayArtifact("replay", num_queries, query_len, decode, row, trace)


def microbench_artifact(
    *,
    op_name: str,
    shape: tuple[int, ...],
    dtype: str,
    replay_row: dict[str, object] | None = None,
    **rejected: object,
) -> MicrobenchArtifact:
    """One microbench request described by op, shape, and dtype."""
    _reject_identity(rejected)
    if not op_name or not shape or any(dim < 1 for dim in shape):
        raise ValueError("microbench case needs an op name and a positive shape")
    shape_text = ",".join(str(dim) for dim in shape)
    row = replay_row or {
        "Input Shapes": shape_text,
        "Input Data Types": dtype,
        "Input Formats": "ND",
        "Output Shapes": shape_text,
        "Output Data Types": dtype,
        "Output Formats": "ND",
    }
    request = {"schema_version": 1, "kernel_type": op_name, "replay_row": row}
    return MicrobenchArtifact("microbench", op_name, shape, dtype, request)


@dataclass(frozen=True)
class L2BuilderContract:
    """Public interface and result type for one canonical L2 builder."""

    kind: str
    builder: object
    artifact_type: type
    required_parameters: tuple[str, ...]
    optional_parameters: tuple[str, ...] = ()


L2_CONTRACTS = (
    L2BuilderContract(
        "quant",
        quant_artifact,
        QuantArtifact,
        ("linear_action",),
        ("attention_action", "input_shape", "output_features"),
    ),
    L2BuilderContract(
        "parallel",
        parallel_artifact,
        ParallelArtifact,
        ("world_size", "tensor_parallel"),
        ("expert_parallel",),
    ),
    L2BuilderContract("fx", fx_artifact, FxArtifact, (), ("trace_func", "trace_args")),
    L2BuilderContract(
        "runtime_event",
        runtime_event_artifact,
        RuntimeEventArtifact,
        ("op_name", "duration_us"),
        ("args", "kwargs", "out"),
    ),
    L2BuilderContract(
        "diagnostics",
        diagnostics_artifact,
        DiagnosticsArtifact,
        ("region_id", "status"),
        (
            "layer_index",
            "stage_id",
            "rule_id",
            "comparison_kind",
            "message",
            "expected",
            "actual",
            "left_evidence",
            "right_evidence",
        ),
    ),
    L2BuilderContract("replay", replay_artifact, ReplayArtifact, ("num_queries", "query_len"), ("decode", "fields")),
    L2BuilderContract(
        "microbench",
        microbench_artifact,
        MicrobenchArtifact,
        ("op_name", "shape", "dtype"),
        ("replay_row",),
    ),
)

L2_BUILDERS = {contract.kind: contract.builder for contract in L2_CONTRACTS}

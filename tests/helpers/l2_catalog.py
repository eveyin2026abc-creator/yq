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

"""Executable catalog of the seven L2 artifact builders.

Each row is one accepted sample or one rejected sample. The catalog is the
contract for synthetic L2: no row carries a model identity, and a rejected row
names the error the builder must raise.
"""

from __future__ import annotations

from dataclasses import dataclass

from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction

from tests.helpers.l2_artifacts import (
    L2_KINDS,
    diagnostics_artifact,
    fx_artifact,
    microbench_artifact,
    parallel_artifact,
    quant_artifact,
    replay_artifact,
    runtime_event_artifact,
)

BUILDERS = {
    "quant": quant_artifact,
    "parallel": parallel_artifact,
    "fx": fx_artifact,
    "runtime_event": runtime_event_artifact,
    "diagnostics": diagnostics_artifact,
    "replay": replay_artifact,
    "microbench": microbench_artifact,
}


@dataclass(frozen=True)
class CatalogCase:
    kind: str
    name: str
    valid: bool
    kwargs: tuple[tuple[str, object], ...] = ()
    error: str = ""

    def arguments(self) -> dict[str, object]:
        return dict(self.kwargs)


def _case(
    kind: str,
    name: str,
    *,
    valid: bool,
    error: str = "",
    **kwargs: object,
) -> CatalogCase:
    return CatalogCase(kind, name, valid, tuple(kwargs.items()), error)


CATALOG: tuple[CatalogCase, ...] = (
    _case("quant", "disabled", valid=True, linear_action=QuantizeLinearAction.DISABLED),
    _case("quant", "w8a8_dynamic", valid=True, linear_action=QuantizeLinearAction.W8A8_DYNAMIC),
    _case("quant", "w8a8_static", valid=True, linear_action=QuantizeLinearAction.W8A8_STATIC),
    _case("quant", "w4a8_dynamic", valid=True, linear_action=QuantizeLinearAction.W4A8_DYNAMIC),
    _case("quant", "fp8", valid=True, linear_action=QuantizeLinearAction.FP8),
    _case("quant", "mxfp4", valid=True, linear_action=QuantizeLinearAction.MXFP4),
    _case(
        "quant",
        "attention_int8",
        valid=True,
        linear_action=QuantizeLinearAction.DISABLED,
        attention_action=QuantizeAttentionAction.INT8,
    ),
    _case(
        "quant",
        "rejects_model_id",
        valid=False,
        error="model identity",
        linear_action=QuantizeLinearAction.DISABLED,
        model_id="Qwen/Qwen3-32B",
    ),
    _case("parallel", "single_device", valid=True, world_size=1, tensor_parallel=1),
    _case("parallel", "tp2", valid=True, world_size=2, tensor_parallel=2),
    _case("parallel", "tp2_ep2", valid=True, world_size=2, tensor_parallel=2, expert_parallel=2),
    _case("parallel", "tp4_ep2", valid=True, world_size=8, tensor_parallel=4, expert_parallel=2),
    _case("parallel", "ep_only", valid=True, world_size=4, tensor_parallel=1, expert_parallel=4),
    _case("parallel", "world_not_divisible_by_tp", valid=False, error="divisible", world_size=3, tensor_parallel=2),
    _case(
        "parallel",
        "world_not_divisible_by_ep",
        valid=False,
        error="divisible",
        world_size=4,
        tensor_parallel=2,
        expert_parallel=3,
    ),
    _case("parallel", "zero_world", valid=False, error="positive", world_size=0, tensor_parallel=1),
    _case("parallel", "negative_tp", valid=False, error="positive", world_size=2, tensor_parallel=-1),
    _case(
        "parallel",
        "rejects_model_id",
        valid=False,
        error="model identity",
        world_size=1,
        tensor_parallel=1,
        model_id="deepseek",
    ),
    _case("fx", "placeholder_graph", valid=True),
    _case("fx", "rejects_model_id", valid=False, error="model identity", model_id="unused"),
    _case("runtime_event", "zero_duration", valid=True, op_name="aten.mm", duration_us=0),
    _case("runtime_event", "named_event", valid=True, op_name="aten.addmm", duration_us=12),
    _case("runtime_event", "long_event", valid=True, op_name="fused_moe", duration_us=10_000),
    _case("runtime_event", "empty_name", valid=False, error="name", op_name="", duration_us=1),
    _case("runtime_event", "negative_duration", valid=False, error="non-negative", op_name="aten.mm", duration_us=-1),
    _case(
        "runtime_event",
        "rejects_model_id",
        valid=False,
        error="model identity",
        op_name="aten.mm",
        duration_us=1,
        model_id="unused",
    ),
    _case("diagnostics", "pass_language", valid=True, region_id="language", status="PASS"),
    _case("diagnostics", "fail_visual", valid=True, region_id="visual", status="FAIL"),
    _case("diagnostics", "pass_mtp", valid=True, region_id="mtp", status="PASS"),
    _case("diagnostics", "unknown_status", valid=False, error="PASS or FAIL", region_id="language", status="SKIP"),
    _case("diagnostics", "empty_status", valid=False, error="PASS or FAIL", region_id="language", status=""),
    _case(
        "diagnostics",
        "rejects_model_id",
        valid=False,
        error="model identity",
        region_id="language",
        status="PASS",
        model_id="unused",
    ),
    _case("replay", "prefill", valid=True, num_queries=2, query_len=10, decode=False),
    _case("replay", "decode", valid=True, num_queries=1, query_len=1, decode=True),
    _case("replay", "large_batch", valid=True, num_queries=32, query_len=128),
    _case("replay", "zero_batch", valid=False, error="positive", num_queries=0, query_len=10),
    _case("replay", "zero_query", valid=False, error="positive", num_queries=2, query_len=0),
    _case(
        "replay",
        "rejects_model_id",
        valid=False,
        error="model identity",
        num_queries=1,
        query_len=1,
        model_id="unused",
    ),
    _case("microbench", "square_fp16", valid=True, op_name="aten.mm", shape=(128, 128), dtype="fp16"),
    _case("microbench", "moe_int8", valid=True, op_name="grouped_matmul", shape=(8, 256, 256), dtype="int8"),
    _case("microbench", "vector", valid=True, op_name="aten.silu", shape=(1024,), dtype="fp32"),
    _case("microbench", "empty_name", valid=False, error="op name", op_name="", shape=(8,), dtype="fp16"),
    _case("microbench", "empty_shape", valid=False, error="positive shape", op_name="aten.mm", shape=(), dtype="fp16"),
    _case(
        "microbench",
        "zero_dim",
        valid=False,
        error="positive shape",
        op_name="aten.mm",
        shape=(0, 16),
        dtype="fp16",
    ),
    _case(
        "microbench",
        "rejects_model_id",
        valid=False,
        error="model identity",
        op_name="aten.mm",
        shape=(8, 8),
        dtype="fp16",
        model_id="unused",
    ),
)


def catalog_kinds() -> tuple[str, ...]:
    return tuple(dict.fromkeys(case.kind for case in CATALOG))


def cases_for(kind: str) -> tuple[CatalogCase, ...]:
    return tuple(case for case in CATALOG if case.kind == kind)


def run_catalog_case(case: CatalogCase):
    """Build one catalog row, or raise the builder's error for a rejected row."""

    builder = BUILDERS[case.kind]
    if not case.valid:
        try:
            builder(**case.arguments())
        except (TypeError, ValueError):
            raise
        else:
            raise AssertionError(f"{case.kind}:{case.name} was expected to fail")
    artifact = builder(**case.arguments())
    if artifact.kind != case.kind:
        raise AssertionError(f"{case.kind}:{case.name} returned kind {artifact.kind}")
    return artifact


def catalog_covers_kinds() -> bool:
    return catalog_kinds() == L2_KINDS

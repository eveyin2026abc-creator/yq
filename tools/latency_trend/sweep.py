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

"""Sweep batch size on one loaded model and collect network and operator latency."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from tools.latency_trend.report import BoundShare, LatencyReport, SweepPoint, aggregate_operators

logger = logging.getLogger(__name__)


@dataclass
class SweepConfig:
    model_id: str
    batch_sizes: Sequence[int]
    device: str = "TEST_DEVICE"
    phase: str = "decode"
    query_length: int = 1
    context_length: int = 4096
    tp_size: int = 1
    quantize_linear: str = "DISABLED"
    performance_model: str = "analytic"
    top_k: int = 10
    num_hidden_layers: int = 0


def _classified_share(breakdowns: dict[str, dict[str, float]], model_name: str) -> BoundShare:
    key = f"{model_name}_OpBound"
    values = breakdowns.get(key)
    if values is None:
        for name, candidate in breakdowns.items():
            if name.endswith("_OpBound"):
                values = candidate
                break
    return BoundShare.from_mapping(values or {})


def _component_share(events: Sequence[Any], model_name: str) -> BoundShare:
    total = BoundShare()
    for event in events:
        if getattr(event, "is_spill", False):
            continue
        result = event.perf_results.get(model_name)
        if result is None:
            continue
        total = total.add(BoundShare.from_result(result))
    return total


def point_from_runtime(
    batch_size: int,
    execution_time_s: float,
    breakdowns: dict[str, dict[str, float]],
    events: Sequence[Any],
    model_name: str,
) -> SweepPoint:
    """Build one sweep point from a finished runtime."""
    return SweepPoint(
        batch_size=batch_size,
        latency_s=execution_time_s,
        classified=_classified_share(breakdowns, model_name),
        components=_component_share(events, model_name),
        operators=aggregate_operators(events, model_name),
    )


def run_sweep(config: SweepConfig) -> LatencyReport:
    """Load ``config.model_id`` once and resimulate each batch size."""
    if config.phase not in {"decode", "prefill"}:
        raise ValueError(f"phase must be decode or prefill, got {config.phase!r}")
    batch_sizes = [int(size) for size in config.batch_sizes]
    if not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("batch_sizes must be positive integers")

    import tensor_cast.device_profiles  # noqa: F401  (registers built-in profiles)
    from tensor_cast.core.input_generator import generate_inputs
    from tensor_cast.core.model_runner import ModelRunner
    from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
    from tensor_cast.core.user_config import UserInputConfig

    quantize = QuantizeLinearAction(config.quantize_linear)
    user_input = UserInputConfig(
        device=config.device,
        model_id=config.model_id,
        num_queries=batch_sizes[0],
        query_len=config.query_length,
        context_length=config.context_length,
        decode=config.phase == "decode",
        tp_size=config.tp_size,
        world_size=config.tp_size,
        quantize_linear_action=quantize,
        performance_model=config.performance_model,
        do_compile=False,
        num_hidden_layers_override=config.num_hidden_layers,
    )
    logger.info("Building %s once for %d batch sizes", user_input.model_id, len(batch_sizes))
    runner = ModelRunner(user_input)
    points: list[SweepPoint] = []
    for batch_size in batch_sizes:
        user_input.num_queries = batch_size
        runner.request_info_default = [user_input.get_request_info()]
        captured: dict[str, Any] = {}

        def _observe(runtime, captured=captured):
            captured["runtime"] = runtime

        logger.info("Simulating batch_size=%s", batch_size)
        metrics = runner.run_inference(generate_inputs_func=generate_inputs, runtime_observer=_observe)
        runtime = captured.get("runtime")
        if runtime is None:
            raise RuntimeError(f"batch {batch_size} finished without a runtime")
        model_name = metrics.perf_model_name or config.performance_model
        latency_s = float(metrics.execution_time_s.get(model_name, 0.0))
        point = point_from_runtime(
            batch_size,
            latency_s,
            metrics.breakdowns,
            runtime.event_list,
            model_name,
        )
        logger.info("batch %s latency %.4f ms bound %s", batch_size, latency_s * 1e3, point.bound)
        points.append(point)

    return LatencyReport(
        model_id=user_input.model_id,
        device=config.device,
        phase=config.phase,
        query_length=config.query_length,
        context_length=config.context_length,
        tp_size=config.tp_size,
        quantize_linear=config.quantize_linear,
        performance_model=config.performance_model,
        points=points,
        top_k=config.top_k,
    )

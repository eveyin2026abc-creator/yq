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

"""Latency trend aggregation and HTML report."""

from __future__ import annotations

from types import SimpleNamespace

from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.performance_model.bound_analyzer import COMPUTE_BOUND_MMA, MEMORY_BOUND

from cli.main import main
from tests.helpers.cli_runner import run_cli_main
from tools.latency_trend.cli import config_from_args, build_parser
from tools.latency_trend.report import (
    BoundShare,
    LatencyReport,
    OperatorSample,
    SweepPoint,
    render_html,
)
from tools.latency_trend.sweep import point_from_runtime


def _sample(name: str, latency_s: float, memory_s: float, mma_s: float) -> OperatorSample:
    return OperatorSample(name, latency_s, 2, BoundShare(memory_s=memory_s, mma_s=mma_s))


def _point(batch: int, memory_s: float, mma_s: float) -> SweepPoint:
    share = BoundShare(memory_s=memory_s, mma_s=mma_s)
    return SweepPoint(
        batch_size=batch,
        latency_s=memory_s + mma_s,
        classified=share,
        components=share,
        operators=(
            _sample("aten.mm", mma_s, 0.0, mma_s),
            _sample("aten.add", memory_s, memory_s, 0.0),
        ),
    )


def _report() -> LatencyReport:
    return LatencyReport(
        model_id="Qwen/Qwen3-0.6B",
        device="TEST_DEVICE",
        phase="decode",
        query_length=1,
        context_length=4096,
        tp_size=1,
        quantize_linear="DISABLED",
        performance_model="analytic",
        points=[_point(1, 0.004, 0.001), _point(8, 0.001, 0.006)],
        top_k=10,
    )


def test_network_bound_switches_from_memory_to_compute() -> None:
    report = _report()

    assert report.points[0].bound == MEMORY_BOUND
    assert report.points[1].bound == COMPUTE_BOUND_MMA
    transitions = report.network_transitions()
    assert len(transitions) == 1
    assert transitions[0].batch_size == 8
    assert transitions[0].previous == MEMORY_BOUND
    assert transitions[0].current == COMPUTE_BOUND_MMA
    assert report.top_operators()[0] == "aten.mm"


def test_html_marks_the_bound_change_and_top_operators() -> None:
    page = render_html(_report())

    assert "从 batch 8 起转为矩阵计算 compute" in page
    assert "算子时延趋势" in page
    assert "id='op-tags'" in page
    assert "data-op='*'" in page
    assert "data-op='aten.mm'" in page
    assert "id='point-readout'" in page
    assert "class='point'" in page
    assert "占整网" in page
    assert "id='download-report'" in page
    assert page.count("<polyline") >= 3
    assert "aten.mm" in page
    assert "aten.add" in page
    assert "memory →" not in page or "访存 memory → 矩阵计算 compute" in page
    assert "Qwen/Qwen3-0.6B" in page


def test_point_from_runtime_sums_operator_components() -> None:
    memory = PerformanceModel.Result(0.002, {"memory_access_time_s": 0.002})
    compute = PerformanceModel.Result(0.005, {"mma_ops_time_s": 0.005, "compute_time_s": 0.005})
    events = [
        SimpleNamespace(
            is_spill=False,
            perf_results={"analytic": memory},
            op_invoke_info=SimpleNamespace(func="aten.add"),
        ),
        SimpleNamespace(
            is_spill=False,
            perf_results={"analytic": compute},
            op_invoke_info=SimpleNamespace(func="aten.mm"),
        ),
        SimpleNamespace(
            is_spill=True,
            perf_results={"analytic": compute},
            op_invoke_info=SimpleNamespace(func="spill"),
        ),
    ]

    point = point_from_runtime(
        4,
        0.006,
        {"analytic_OpBound": {MEMORY_BOUND: 0.002, COMPUTE_BOUND_MMA: 0.005}},
        events,
        "analytic",
    )

    assert point.bound == COMPUTE_BOUND_MMA
    assert [sample.name for sample in point.operators] == ["aten.mm", "aten.add"]
    assert point.operators[0].calls == 1
    assert point.operators[0].bound == COMPUTE_BOUND_MMA


def test_decode_defaults_and_explicit_lengths() -> None:
    parser = build_parser()
    decode = config_from_args(parser.parse_args(["org/model"]))
    prefill = config_from_args(parser.parse_args(["org/model", "--phase", "prefill", "--query-length", "128"]))

    assert decode.phase == "decode"
    assert decode.query_length == 1
    assert decode.context_length == 4096
    assert decode.batch_sizes == [1, 2, 4, 8, 16, 32, 64]
    assert prefill.query_length == 128
    assert prefill.context_length == 0


def test_latency_trend_help_lists_batch_sweep() -> None:
    result = run_cli_main(main, ["inference", "latency-trend", "--help"], prog="msmodeling")

    assert result.returncode == 0
    assert "--batch-sizes" in result.stdout
    assert "--phase" in result.stdout
    assert "--output" in result.stdout

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


def test_bound_follows_is_compute_bound() -> None:
    tied = BoundShare(memory_s=0.004, mma_s=0.004, communication_s=0.010)
    compute = BoundShare(memory_s=0.004, mma_s=0.005, communication_s=0.020)
    gp = BoundShare(memory_s=0.001, mma_s=0.002, gp_s=0.003)

    assert tied.dominant() == MEMORY_BOUND
    assert compute.dominant() == COMPUTE_BOUND_MMA
    assert gp.dominant() == COMPUTE_BOUND_MMA
    point = SweepPoint(
        batch_size=1,
        latency_s=0.029,
        classified=BoundShare(communication_s=0.020),
        components=compute,
    )
    assert point.bound == COMPUTE_BOUND_MMA


def test_network_bound_switches_from_memory_to_compute() -> None:
    report = _report()

    assert report.points[0].bound == MEMORY_BOUND
    assert report.points[1].bound == COMPUTE_BOUND_MMA
    transitions = report.network_transitions()
    assert len(transitions) == 1
    assert transitions[0].batch_size == 8
    assert transitions[0].previous == MEMORY_BOUND
    assert transitions[0].current == COMPUTE_BOUND_MMA
    assert report.top_operators() == ["aten.mm", "aten.add"]


def test_operator_selection_keeps_ten_percent_share_or_top_ten() -> None:
    def point(values: tuple[tuple[str, float], ...]) -> SweepPoint:
        operators = tuple(OperatorSample(name, latency, 1, BoundShare(memory_s=latency)) for name, latency in values)
        return SweepPoint(
            batch_size=1,
            latency_s=1.0,
            classified=BoundShare(memory_s=1.0),
            components=BoundShare(memory_s=1.0),
            operators=operators,
        )

    report = _report()
    report.points = [point(tuple((f"op{index}", 0.10) for index in range(12)))]
    assert len(report.top_operators()) == 12

    mixed = (("a", 0.20), ("b", 0.20), ("c", 0.20)) + tuple((f"t{index}", 0.01) for index in range(10))
    report.points = [point(mixed)]
    selected = report.top_operators()
    assert selected[:3] == ["a", "b", "c"]
    assert len(selected) == 10

    # A shape can be 10% of a small batch and still sit outside the total ranking.
    report.points = [
        point((("heavy", 0.50),) + tuple((f"t{index}", 0.04) for index in range(12))),
        SweepPoint(
            batch_size=2,
            latency_s=0.10,
            classified=BoundShare(memory_s=0.10),
            components=BoundShare(memory_s=0.10),
            operators=(OperatorSample("spike", 0.02, 1, BoundShare(memory_s=0.02)),),
        ),
    ]
    selected = report.top_operators()
    assert "spike" in selected
    assert len(selected) >= 10


def test_network_chart_marks_out_of_memory() -> None:
    report = _report()
    report.points.append(_point(16, 0.002, 0.002))
    report.points[-1].oom = True
    report.points[-1].memory_available_gb = -1.5

    page = render_html(report)

    assert "class='oom-region'" in page
    assert "class='oom-boundary'" in page
    assert "class='point network-point oom-point'" in page
    assert "class='oom first-oom-label'>OOM</text>" in page
    assert "batch 16 显存不足" in page
    assert "可用显存 -1.50 GB" in page


def test_html_marks_the_bound_change_and_top_operators() -> None:
    page = render_html(_report())

    assert "从 batch 8 起转为计算 compute" in page
    assert "算子时延" in page
    assert "id='op-tags'" in page
    assert "data-op='*'" in page
    assert "data-op='aten.mm'" in page
    assert "id='point-readout'" in page
    assert "class='point'" in page
    assert "占整网" in page
    assert "id='download-report'" in page
    report = _report()
    report.command = "msmodeling inference text-generate MODEL \\\n  --num-queries 1 2 4 8"
    page_with_command = render_html(report)
    assert "id='copy-command'" in page_with_command
    assert "id='command-text'" in page_with_command
    assert "--num-queries 1 2 4 8" in page_with_command
    assert page.count("<polyline") >= 2
    assert "占整网达到 10%" in page
    assert "aten.mm" in page
    assert "memory →" not in page or "访存 memory → 计算 compute" in page
    assert "Qwen/Qwen3-0.6B" in page


def test_same_shape_uses_analytic_average() -> None:
    import torch

    slow = PerformanceModel.Result(0.004, {"memory_access_time_s": 0.004})
    fast = PerformanceModel.Result(0.002, {"memory_access_time_s": 0.002})
    other = PerformanceModel.Result(0.009, {"memory_access_time_s": 0.009})
    narrow = torch.empty(2, 4)
    wide = torch.empty(8, 4)
    events = [
        SimpleNamespace(
            is_spill=False,
            perf_results={"analytic": slow},
            op_invoke_info=SimpleNamespace(func="aten.add", args=(narrow,)),
        ),
        SimpleNamespace(
            is_spill=False,
            perf_results={"analytic": fast},
            op_invoke_info=SimpleNamespace(func="aten.add", args=(narrow,)),
        ),
        SimpleNamespace(
            is_spill=False,
            perf_results={"analytic": other},
            op_invoke_info=SimpleNamespace(func="aten.add", args=(wide,)),
        ),
    ]

    point = point_from_runtime(4, 0.015, {}, events, "analytic")

    by_shape = {sample.input_shapes: sample for sample in point.operators}
    assert by_shape["[2, 4]"].latency_s == 0.003
    assert by_shape["[2, 4]"].calls == 2
    assert by_shape["[8, 4]"].latency_s == 0.009
    assert by_shape["[8, 4]"].calls == 1
    assert "aten.add [8, 4]" in render_html(
        LatencyReport(
            model_id="shape",
            device="TEST_DEVICE",
            phase="decode",
            query_length=1,
            context_length=0,
            tp_size=1,
            quantize_linear="DISABLED",
            performance_model="analytic",
            points=[point],
        )
    )


def test_batch_dimension_keeps_one_operator_series() -> None:
    import torch

    result = PerformanceModel.Result(0.002, {"memory_access_time_s": 0.002})
    points = []
    for batch in (2, 8):
        tensor = torch.empty(batch, 4)
        points.append(
            point_from_runtime(
                batch,
                0.002,
                {},
                [
                    SimpleNamespace(
                        is_spill=False,
                        perf_results={"analytic": result},
                        op_invoke_info=SimpleNamespace(func="aten.add", args=(tensor,)),
                    )
                ],
                "analytic",
            )
        )

    assert points[0].operators[0].name == "aten.add [B, 4]"
    assert points[1].operators[0].name == "aten.add [B, 4]"
    assert points[0].operators[0].latency_s == 0.002


def _attention_events(batch: int, blocks: int, max_blocks: int, query_tokens: int | None = None):
    import torch

    tokens = batch if query_tokens is None else query_tokens
    result = PerformanceModel.Result(0.004, {"memory_access_time_s": 0.004})
    args = (
        torch.empty(tokens, 2048),
        torch.empty(blocks, 128, 8, 128),
        torch.empty(blocks, 128, 8, 128),
        torch.empty(batch, max_blocks),
        torch.empty(batch + 1),
        torch.empty(batch),
        torch.empty(batch),
    )
    return [
        SimpleNamespace(
            is_spill=False,
            perf_results={"analytic": result},
            op_invoke_info=SimpleNamespace(func="tensor_cast.attention.default", args=args),
        )
    ]


def test_attention_cache_capacity_stays_one_shape() -> None:
    short = point_from_runtime(16, 0.02, {}, _attention_events(16, 513, 33), "analytic")
    long = point_from_runtime(128, 0.08, {}, _attention_events(128, 4098, 33), "analytic")
    other_seq = point_from_runtime(16, 0.02, {}, _attention_events(16, 1026, 65), "analytic")
    prefill_small = point_from_runtime(4, 0.02, {}, _attention_events(4, 17, 4, query_tokens=4 * 512), "analytic")
    prefill_large = point_from_runtime(16, 0.04, {}, _attention_events(16, 65, 4, query_tokens=16 * 512), "analytic")

    expected = (
        "tensor_cast.attention.default [B, 2048], [cache, 128, 8, 128], [cache, 128, 8, 128], [B, 33], [B], [B], [B]"
    )
    assert short.operators[0].name == expected
    assert long.operators[0].name == expected
    assert "[B, 65]" in other_seq.operators[0].name
    assert prefill_small.operators[0].name == prefill_large.operators[0].name
    assert "[B*512, 2048]" in prefill_small.operators[0].name


def _matmul_events(batch: int, rows: int, weight: tuple[int, int]):
    import torch

    result = PerformanceModel.Result(0.004, {"mma_ops_time_s": 0.004, "compute_time_s": 0.004})
    activation = torch.empty(rows, weight[0])
    matrix = torch.empty(*weight)
    return [
        SimpleNamespace(
            is_spill=False,
            perf_results={"analytic": result},
            op_invoke_info=SimpleNamespace(func="aten.mm.default", args=(activation, matrix)),
        )
    ]


def test_prefill_matmul_keeps_one_curve_per_weight() -> None:
    small = point_from_runtime(4, 0.02, {}, _matmul_events(4, 4 * 512, (5120, 25600)), "analytic")
    large = point_from_runtime(32, 0.08, {}, _matmul_events(32, 32 * 512, (5120, 25600)), "analytic")
    other = point_from_runtime(32, 0.08, {}, _matmul_events(32, 32 * 512, (5120, 8192)), "analytic")
    decode = point_from_runtime(32, 0.02, {}, _matmul_events(32, 32, (1024, 3072)), "analytic")

    expected = "aten.mm.default [B*512, 5120], [5120, 25600]"
    assert small.operators[0].name == expected
    assert large.operators[0].name == expected
    assert other.operators[0].name == "aten.mm.default [B*512, 5120], [5120, 8192]"
    assert decode.operators[0].name == "aten.mm.default [B, 1024], [1024, 3072]"


def test_routed_matmul_merges_by_weight_for_any_model() -> None:
    routed = [
        point_from_runtime(1, 0.01, {}, _matmul_events(1, 1, (2048, 768)), "analytic"),
        point_from_runtime(8, 0.02, {}, _matmul_events(8, 1, (2048, 768)), "analytic"),
        point_from_runtime(128, 0.08, {}, _matmul_events(128, 8, (2048, 768)), "analytic"),
    ]
    dense = [
        point_from_runtime(8, 0.01, {}, _matmul_events(8, 8, (1024, 3072)), "analytic"),
        point_from_runtime(32, 0.04, {}, _matmul_events(32, 32, (1024, 3072)), "analytic"),
    ]
    other = point_from_runtime(128, 0.08, {}, _matmul_events(128, 8, (768, 2048)), "analytic")
    routed[-1].operators = (*routed[-1].operators, *other.operators)

    routed_report = LatencyReport(
        model_id="moe",
        device="TEST_DEVICE",
        phase="decode",
        query_length=1,
        context_length=4096,
        tp_size=1,
        quantize_linear="DISABLED",
        performance_model="analytic",
        points=routed,
    )
    dense_report = LatencyReport(
        model_id="dense",
        device="TEST_DEVICE",
        phase="decode",
        query_length=1,
        context_length=4096,
        tp_size=1,
        quantize_linear="DISABLED",
        performance_model="analytic",
        points=dense,
    )

    routed_name = "aten.mm.default [tokens, 2048], [2048, 768]"
    down_name = "aten.mm.default [tokens, 768], [768, 2048]"
    assert routed_name in routed_report.top_operators()
    assert down_name in routed_report.top_operators()
    assert {sample.name for point in routed for sample in point.operators} == {routed_name, down_name}
    assert dense_report.top_operators() == ["aten.mm.default [B, 1024], [1024, 3072]"]


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


def test_text_generate_command_groups_each_flag() -> None:
    from cli.inference.text_generate import format_text_generate_command

    command = format_text_generate_command(
        [
            "MODEL",
            "--num-queries",
            "1",
            "2",
            "4",
            "8",
            "--query-length",
            "1",
            "--context-length",
            "4096",
            "--decode",
            "--device",
            "TEST_DEVICE",
            "--dump-op-bound-results",
            "--html-file",
            "latency-trend.html",
            "--html-port",
            "8771",
        ]
    )

    assert command.splitlines() == [
        "msmodeling inference text-generate MODEL \\",
        "  --num-queries 1 2 4 8 \\",
        "  --query-length 1 \\",
        "  --context-length 4096 \\",
        "  --decode \\",
        "  --device TEST_DEVICE \\",
        "  --dump-op-bound-results \\",
        "  --html-file latency-trend.html \\",
        "  --html-port 8771",
    ]


def test_output_paths_stay_unchanged_for_one_batch() -> None:
    from cli.inference.text_generate import _scoped_output_path

    assert _scoped_output_path("trace.json", 4, False) == "trace.json"
    assert _scoped_output_path(None, 4, True) is None
    assert _scoped_output_path("trace.json", 4, True) == "trace.batch4.json"
    assert _scoped_output_path("dir/metrics.json", 32, True) == "dir/metrics.batch32.json"


def test_text_generate_one_num_queries_stays_a_single_batch() -> None:
    from cli.inference.text_generate import arg_parse

    args = arg_parse(["--num-queries", "4", "--query-length", "8", "Qwen/Qwen3-32B", "--device=TEST_DEVICE"])

    assert args.num_queries == 4
    assert args.num_query_sweep == [4]
    assert args.html_file is None
    assert args.html_port is None
    assert args.html_bind == "127.0.0.1"


def test_text_generate_num_queries_list_writes_a_trend() -> None:
    import pytest
    from cli.inference.text_generate import arg_parse

    base = ["--query-length", "1", "--context-length", "4096", "--decode", "Qwen/Qwen3-32B", "--device=TEST_DEVICE"]
    with pytest.raises(SystemExit):
        arg_parse(["--num-queries", "1", "2", "4", "8", *base])

    args = arg_parse(
        ["--num-queries", "1", "2", "4", "8", *base, "--dump-op-bound-results", "--html-file", "latency-trend.html"]
    )

    assert args.num_queries == 1
    assert args.num_query_sweep == [1, 2, 4, 8]
    assert args.html_file == "latency-trend.html"
    assert args.html_port is None
    assert args.html_bind == "127.0.0.1"
    assert args.dump_op_bound_results is True
    assert args.decode is True


def test_html_port_requires_html_file_and_defaults_to_localhost() -> None:
    import pytest
    from cli.inference.text_generate import arg_parse

    base = ["--num-queries", "1", "--query-length", "1", "Qwen/Qwen3-32B", "--device=TEST_DEVICE"]
    with pytest.raises(SystemExit):
        arg_parse([*base, "--html-port", "8771"])

    args = arg_parse([*base, "--html-file", "out.html", "--html-port", "8771"])

    assert args.html_port == 8771
    assert args.html_bind == "127.0.0.1"


def test_html_server_binds_localhost_and_serves_the_file(tmp_path) -> None:
    import threading
    import urllib.request

    from cli.inference.text_generate import _html_server

    page = tmp_path / "report.html"
    page.write_text("latency-report", encoding="utf-8")
    httpd = _html_server(tmp_path, "127.0.0.1", 0)
    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        assert host == "127.0.0.1"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/report.html", timeout=5) as response:
            assert response.read() == b"latency-report"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

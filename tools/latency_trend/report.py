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

"""Aggregate analytic bound components and render a self-contained HTML report."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch

from tensor_cast.performance_model.base import PerformanceModel
from tensor_cast.performance_model.bound_analyzer import (
    COMMUNICATION_BOUND,
    COMPUTE_BOUND_GP,
    COMPUTE_BOUND_MMA,
    MEMORY_BOUND,
    UNKNOWN_BOUND,
    BoundAnalyzer,
)

BOUND_ORDER = (
    MEMORY_BOUND,
    COMPUTE_BOUND_MMA,
    UNKNOWN_BOUND,
)

BOUND_LABELS = {
    MEMORY_BOUND: "访存 memory",
    COMMUNICATION_BOUND: "通信 communication",
    COMPUTE_BOUND_MMA: "计算 compute",
    COMPUTE_BOUND_GP: "计算 compute",
    UNKNOWN_BOUND: "未分类",
}

BOUND_COLORS = {
    MEMORY_BOUND: "#c47b2b",
    COMMUNICATION_BOUND: "#6d5bd0",
    COMPUTE_BOUND_MMA: "#0f6e8c",
    COMPUTE_BOUND_GP: "#0f6e8c",
    UNKNOWN_BOUND: "#8a8175",
}

SERIES_COLORS = (
    "#1c1915",
    "#0f6e8c",
    "#c47b2b",
    "#1d7a46",
    "#6d5bd0",
    "#8d2f2f",
    "#3f7cac",
    "#b85c38",
    "#2f6f4e",
    "#7a4e8a",
)

_COMPONENT_STATS = {
    "memory": "memory_access_time_s",
    "communication": "comm_time_s",
    "mma": "mma_ops_time_s",
    "gp": "gp_ops_time_s",
}


@dataclass
class BoundShare:
    """Seconds attributed to each roofline component."""

    memory_s: float = 0.0
    communication_s: float = 0.0
    mma_s: float = 0.0
    gp_s: float = 0.0

    def total_s(self) -> float:
        return self.memory_s + self.communication_s + self.mma_s + self.gp_s

    def add(self, other: "BoundShare") -> "BoundShare":
        return BoundShare(
            memory_s=self.memory_s + other.memory_s,
            communication_s=self.communication_s + other.communication_s,
            mma_s=self.mma_s + other.mma_s,
            gp_s=self.gp_s + other.gp_s,
        )

    def percent(self) -> dict[str, float]:
        total = self.total_s()
        raw = {
            MEMORY_BOUND: self.memory_s,
            COMMUNICATION_BOUND: self.communication_s,
            COMPUTE_BOUND_MMA: self.mma_s,
            COMPUTE_BOUND_GP: self.gp_s,
        }
        if total <= 0:
            return {key: 0.0 for key in raw}
        return {key: value * 100.0 / total for key, value in raw.items()}

    def dominant(self) -> str:
        """Match trace ``is_compute_bound``: compute_time_s > memory_access_time_s."""
        compute_s = self.mma_s + self.gp_s
        if compute_s <= 0 and self.memory_s <= 0 and self.communication_s <= 0:
            return UNKNOWN_BOUND
        if compute_s > self.memory_s:
            return COMPUTE_BOUND_MMA
        return MEMORY_BOUND

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "BoundShare":
        return cls(
            memory_s=float(values.get(MEMORY_BOUND, 0.0) or 0.0),
            communication_s=float(values.get(COMMUNICATION_BOUND, 0.0) or 0.0),
            mma_s=float(values.get(COMPUTE_BOUND_MMA, 0.0) or 0.0),
            gp_s=float(values.get(COMPUTE_BOUND_GP, 0.0) or 0.0),
        )

    @classmethod
    def from_result(cls, result: PerformanceModel.Result) -> "BoundShare":
        components = BoundAnalyzer.components(result)
        return cls(
            memory_s=components.memory_time_s,
            communication_s=components.communication_time_s,
            mma_s=components.mma_ops_time_s,
            gp_s=components.gp_ops_time_s,
        )


@dataclass
class OperatorSample:
    """One operator at one input shape. ``latency_s`` is the analytic average."""

    name: str
    latency_s: float
    calls: int
    share: BoundShare
    input_shapes: str = ""
    op_name: str = ""

    @property
    def bound(self) -> str:
        return self.share.dominant()

    @property
    def total_s(self) -> float:
        return self.latency_s * self.calls


@dataclass
class SweepPoint:
    batch_size: int
    latency_s: Optional[float]
    classified: BoundShare
    components: BoundShare
    operators: tuple[OperatorSample, ...] = ()
    memory_available_gb: Optional[float] = None
    oom: bool = False

    @property
    def bound(self) -> str:
        # Raw component sums support compute_time_s > memory_access_time_s.
        # Classified OpBound buckets are a different grouping and stay on the mix chart.
        if self.components.total_s() > 0:
            return self.components.dominant()
        return self.classified.dominant()


@dataclass(frozen=True)
class BoundTransition:
    batch_size: int
    previous: str
    current: str


@dataclass
class LatencyReport:
    model_id: str
    device: str
    phase: str
    query_length: int
    context_length: int
    tp_size: int
    quantize_linear: str
    performance_model: str
    points: list[SweepPoint] = field(default_factory=list)
    top_k: int = 10
    command: str = ""

    def network_transitions(self) -> list[BoundTransition]:
        return bound_transitions(self.points, lambda point: point.bound)

    def top_operators(self) -> list[str]:
        unify_matmul_curves(self.points)
        return select_top_operators(self.points, self.top_k)


def bound_transitions(points: Sequence[Any], bound_of, batch_of=None) -> list[BoundTransition]:
    """Record the batch size where the dominant bound changes."""
    if batch_of is None:
        batch_of = lambda point: point.batch_size
    changes: list[BoundTransition] = []
    previous: Optional[str] = None
    for point in points:
        current = bound_of(point)
        if previous is not None and current != previous:
            changes.append(BoundTransition(batch_size=batch_of(point), previous=previous, current=current))
        previous = current
    return changes


def select_top_operators(points: Sequence[SweepPoint], top_k: int) -> list[str]:
    """Keep Top K groups, extending the list for groups reaching 10% at any batch."""
    totals: dict[str, float] = {}
    significant: set[str] = set()
    for point in points:
        for operator in point.operators:
            totals[operator.name] = totals.get(operator.name, 0.0) + operator.total_s
            if point.latency_s and operator.total_s / point.latency_s >= 0.10:
                significant.add(operator.name)
    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return [
        name
        for index, (name, _total) in enumerate(ranked)
        if index < top_k or name in significant
    ]


def input_shapes_label(op_info: Any) -> str:
    """Tensor shapes in the same text as Runtime.table_averages(group_by_input_shapes=True)."""
    shapes = []
    for arg in getattr(op_info, "args", ()) or ():
        if isinstance(arg, torch.Tensor):
            shapes.append(str(list(arg.shape)))
    return ", ".join(shapes)


def operator_series_name(op_name: str, input_shapes: str) -> str:
    if input_shapes:
        return f"{op_name} {input_shapes}"
    return op_name


def _split_shape_pieces(input_shapes: str) -> Optional[list[list[str]]]:
    pieces: list[list[str]] = []
    for piece in input_shapes.split("], "):
        if not piece.endswith("]"):
            piece += "]"
        if not (piece.startswith("[") and piece.endswith("]")):
            return None
        inner = piece[1:-1].strip()
        pieces.append([] if not inner else [item.strip() for item in inner.split(",")])
    return pieces


def _join_shape_pieces(pieces: Sequence[Sequence[str]]) -> str:
    return ", ".join("[" + ", ".join(dims) + "]" for dims in pieces)


def _mark_batch_extent(dims: list[str], batch_size: int, scale_multiples: bool) -> None:
    """Rewrite a leading extent that is only the current batch."""
    if not dims or batch_size < 1:
        return
    if len(dims) == 1 and dims[0] == str(batch_size + 1):
        dims[0] = "B"
        return
    if dims[0] == str(batch_size):
        dims[0] = "B"
        return
    if not scale_multiples or not dims[0].isdigit():
        return
    size = int(dims[0])
    if size > batch_size and size % batch_size == 0:
        dims[0] = f"B*{size // batch_size}"


def _canonicalize_attention_pieces(pieces: list[list[str]], batch_size: int) -> None:
    """Drop batch-sized cache allocation from the attention shape.

    The latency formula uses heads, head size, and per-request query/seq lengths.
    Key and value share the paged cache shape (total_num_blocks, block, heads, dim);
    total_num_blocks is capacity for the whole batch. query_start_loc is batch + 1.
    Neither one is a second attention shape.
    """
    signatures = [tuple(dims) for dims in pieces]
    for dims, signature in zip(pieces, signatures):
        if len(dims) == 4 and signatures.count(signature) >= 2:
            dims[0] = "cache"
            continue
        _mark_batch_extent(dims, batch_size, scale_multiples=True)


def _is_matmul(op_name: str) -> bool:
    """Gemm whose latency is ``2 * M * K * N``. The second tensor is the weight."""
    return "aten.mm" in op_name or "static_quant_linear" in op_name


def _canonicalize_matmul_pieces(pieces: list[list[str]], batch_size: int) -> None:
    """One curve per weight. ``M`` is the batch or ``batch * query`` token count.

    Only the activation's leading dimension is rewritten. A weight such as
    ``[5120, 25600]`` stays numeric even when 5120 is divisible by the batch.
    """
    if pieces:
        _mark_batch_extent(pieces[0], batch_size, scale_multiples=True)


def _matmul_weight_key(op_name: str, input_shapes: str) -> Optional[tuple]:
    """Identity of a matmul without the token count M."""
    if not _is_matmul(op_name):
        return None
    pieces = _split_shape_pieces(input_shapes)
    if not pieces or not pieces[0]:
        return None
    tail = [tuple(dims) for dims in pieces]
    tail[0] = tuple(tail[0][1:])
    return (op_name, tuple(tail))


def _resolved_matmul_m(label: str, batch_size: int) -> Optional[int]:
    if label == "B":
        return batch_size
    if label.startswith("B*") and label[2:].isdigit():
        return batch_size * int(label[2:])
    if label.isdigit():
        return int(label)
    return None


def _stable_matmul_token(observations: Sequence[tuple[int, str]]) -> str:
    """One label for every batch of the same weight.

    ``B`` and ``B*query`` cover dense decode and prefill. Expert routing loads
    a token count that is not a multiple of the batch, so those stay one curve
    under ``tokens`` for any model.
    """
    quotients = []
    for batch_size, label in observations:
        if label == "tokens":
            return "tokens"
        value = _resolved_matmul_m(label, batch_size)
        if value is None or batch_size < 1 or value % batch_size != 0:
            return "tokens"
        quotients.append(value // batch_size)
    if len(set(quotients)) != 1:
        return "tokens"
    quotient = quotients[0]
    return "B" if quotient == 1 else f"B*{quotient}"


def _replace_matmul_m(input_shapes: str, token: str) -> str:
    pieces = _split_shape_pieces(input_shapes)
    if not pieces or not pieces[0]:
        return input_shapes
    pieces[0][0] = token
    return _join_shape_pieces(pieces)


def unify_matmul_curves(points: Sequence[SweepPoint]) -> None:
    """Join matmul calls that share a weight and differ only by token count."""
    grouped: dict[tuple, list[tuple[SweepPoint, OperatorSample]]] = {}
    for point in points:
        for sample in point.operators:
            op_name = sample.op_name or sample.name.split(" ", 1)[0]
            key = _matmul_weight_key(op_name, sample.input_shapes)
            if key is None:
                continue
            grouped.setdefault(key, []).append((point, sample))
    if not grouped:
        return
    tokens = {
        key: _stable_matmul_token(
            [(point.batch_size, _split_shape_pieces(sample.input_shapes)[0][0]) for point, sample in rows]
        )
        for key, rows in grouped.items()
    }
    for point in points:
        merged: dict[tuple, OperatorSample] = {}
        order: list[tuple] = []
        kept: list[OperatorSample] = []
        for sample in point.operators:
            op_name = sample.op_name or sample.name.split(" ", 1)[0]
            key = _matmul_weight_key(op_name, sample.input_shapes)
            if key is None:
                kept.append(sample)
                continue
            shapes = _replace_matmul_m(sample.input_shapes, tokens[key])
            name = operator_series_name(op_name, shapes)
            current = merged.get(key)
            if current is None:
                sample.input_shapes = shapes
                sample.name = name
                sample.op_name = op_name
                merged[key] = sample
                order.append(key)
                continue
            calls = current.calls + sample.calls
            total_s = current.total_s + sample.total_s
            current.calls = calls
            current.latency_s = total_s / calls if calls else 0.0
            current.share = current.share.add(sample.share)
        ordered = [*kept, *(merged[key] for key in order)]
        point.operators = tuple(sorted(ordered, key=lambda item: (-item.total_s, item.name)))


def canonical_input_shapes(input_shapes: str, batch_size: int, op_name: str = "") -> str:
    """Mark a leading dimension equal to this batch as B so the same op stays one curve."""
    if not input_shapes or batch_size < 1:
        return input_shapes
    pieces = _split_shape_pieces(input_shapes)
    if pieces is None:
        return input_shapes
    if "attention" in op_name:
        _canonicalize_attention_pieces(pieces, batch_size)
    elif _is_matmul(op_name):
        _canonicalize_matmul_pieces(pieces, batch_size)
    else:
        for dims in pieces:
            if dims and dims[0] == str(batch_size):
                dims[0] = "B"
    return _join_shape_pieces(pieces)


def aggregate_operators(
    events: Iterable[Any],
    model_name: str,
    batch_size: int = 0,
) -> tuple[OperatorSample, ...]:
    """Average analytic time for each operator name and input shape."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        if getattr(event, "is_spill", False):
            continue
        result = event.perf_results.get(model_name)
        if result is None:
            continue
        op_info = event.op_invoke_info
        op_name = str(op_info.func)
        shapes = canonical_input_shapes(input_shapes_label(op_info), batch_size, op_name)
        slot = grouped.setdefault((op_name, shapes), {"total_s": 0.0, "calls": 0, "share": BoundShare()})
        slot["total_s"] += float(result.execution_time_s)
        slot["calls"] += 1
        slot["share"] = slot["share"].add(BoundShare.from_result(result))
    samples = []
    for (op_name, shapes), slot in grouped.items():
        calls = int(slot["calls"])
        average_s = slot["total_s"] / calls if calls else 0.0
        samples.append(
            OperatorSample(
                name=operator_series_name(op_name, shapes),
                latency_s=average_s,
                calls=calls,
                share=slot["share"],
                input_shapes=shapes,
                op_name=op_name,
            )
        )
    samples.sort(key=lambda sample: (-sample.total_s, sample.name))
    return tuple(samples)


def describe_bounds(points: Sequence[SweepPoint], bound_of, subject: str) -> str:
    if not points:
        return f"{subject}没有采样点。"
    first = bound_of(points[0])
    changes = bound_transitions(points, bound_of)
    if not changes:
        return f"{subject}在扫描的 batch 范围内一直是{BOUND_LABELS[first]}。"
    sentence = f"{subject}在 batch {points[0].batch_size} 是{BOUND_LABELS[first]}"
    for change in changes:
        sentence += f"，从 batch {change.batch_size} 起转为{BOUND_LABELS[change.current]}"
    return sentence + "。"


def report_payload(report: LatencyReport) -> dict[str, Any]:
    """JSON-ready view of the sweep, including per-operator bound labels."""
    top = report.top_operators()
    points = []
    for point in report.points:
        operators = []
        for sample in point.operators:
            if sample.name not in top:
                continue
            operators.append(
                {
                    "name": sample.name,
                    "latency_ms": sample.latency_s * 1e3,
                    "calls": sample.calls,
                    "bound": sample.bound,
                    "components_percent": sample.share.percent(),
                    "input_shapes": sample.input_shapes,
                    "network_share_percent": _network_share(sample.total_s, point.latency_s),
                }
            )
        points.append(
            {
                "batch_size": point.batch_size,
                "latency_ms": None if point.latency_s is None else point.latency_s * 1e3,
                "oom": point.oom,
                "memory_available_gb": point.memory_available_gb,
                "bound": point.bound,
                "classified_percent": point.classified.percent(),
                "components_percent": point.components.percent(),
                "operators": operators,
            }
        )
    return {
        "model_id": report.model_id,
        "device": report.device,
        "phase": report.phase,
        "query_length": report.query_length,
        "context_length": report.context_length,
        "tp_size": report.tp_size,
        "quantize_linear": report.quantize_linear,
        "performance_model": report.performance_model,
        "network_summary": describe_bounds(report.points, lambda point: point.bound, "整网"),
        "transitions": [asdict(item) for item in report.network_transitions()],
        "top_operators": top,
        "command": report.command,
        "points": points,
    }


def render_html(report: LatencyReport, *, download_name: Optional[str] = None) -> str:
    top = report.top_operators()
    short_labels = _operator_labels(top)
    batches = [point.batch_size for point in report.points]
    network_ms = [None if point.latency_s is None else point.latency_s * 1e3 for point in report.points]
    network_bounds = [point.bound for point in report.points]
    filename = download_name or _download_name(report)
    display_model = _display_model_name(report.model_id)
    phase_label = {"decode": "Decode", "prefill": "Prefill"}.get(report.phase, report.phase)
    body = [
        "<header class='page'>",
        "<div class='page-top'>",
        "<div>",
        f"<h1>{html.escape(display_model)} · {html.escape(phase_label)} 时延趋势</h1>",
        "<p class='header-kicker'>Batch sweep · 调度后的设备执行时间</p>",
        "</div>",
        "<p class='toolbar'><button type='button' id='download-report' data-name='"
        + html.escape(filename, quote=True)
        + "'>下载报告</button></p>",
        "</div>",
        _meta_badges(report),
        _command_block(report.command),
        "</header>",
        f"<section><h2>{html.escape(display_model)} · {html.escape(phase_label)} 整网执行时延</h2>",
        "<p class='note'>纵轴是调度后的设备执行时间，单位 ms。</p>",
        "<div class='section-tools'><p id='network-readout' class='section-readout'>"
        "点击曲线圆点或 Batch 读数，查看时延、bound 与显存状态。</p>",
        _legend(),
        "</div>",
        "<div class='network-analysis'>",
        _latency_chart(
            "整网",
            batches,
            network_ms,
            network_bounds,
            width=960,
            height=360,
            oom=[point.oom for point in report.points],
            available_gb=[point.memory_available_gb for point in report.points],
        ),
        _network_batch_panel(report),
        "</div>",
        _oom_note(report),
        _network_table(report),
        "</section>",
        f"<section><h2>{html.escape(display_model)} · {html.escape(phase_label)} 关键算子时延趋势</h2>",
        "<p class='note'>每条曲线代表同一算子、同一输入 shape 的 analytic avg；"
        "随 batch 变化的维度用 B 表示，其余维度相同才连成一条曲线。"
        "Attention 的 KV cache 容量和 query_start_loc 随 batch 分配，记为同一 shape。"
        "Matmul 的 M 等于 batch 或 batch×query 时写成 B 或 B*query，同一块权重连成一条曲线。"
        "专家分到的 token 数不是 batch 的整数倍时，仍按权重合并，M 记为 tokens。"
        "优先展示任一 batch 上占整网达到 10% 的算子；不足 10 个时，按 analytic total 补足。"
        "圆点中的“占整网”为该组 analytic total 与整网调度时间的比值；多流重叠时可超过 100%。</p>",
        "<div class='section-tools'><p id='point-readout' class='section-readout'>"
        "点击算子曲线圆点，查看 batch、analytic avg、bound 与占整网比例。</p>",
        _legend(),
        "</div>",
        "<div class='view-switch'><button type='button' id='toggle-all-names' aria-expanded='false'>展开全部名称</button>"
        "<span class='view-mode' role='group' aria-label='显示方式'>"
        "<button type='button' class='view-button on' data-view='chart'>曲线</button>"
        "<button type='button' class='view-button' data-view='table'>表格</button></span></div>",
        "<div class='op-analysis' id='op-chart-view'>",
        _top_trend_chart(report),
        _operator_tags(report, top),
        "</div>",
        "<div id='op-table-view' hidden>",
        _top_latency_table(report),
        "</div>",
        _operator_detail_panels(report, top, short_labels),
        "</section>",
    ]
    payload = json.dumps(report_payload(report), ensure_ascii=False)
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Latency bound trend</title><style>"
        + _STYLE
        + "</style></head><body>"
        + "".join(body)
        + "<script>"
        + _FILTER_SCRIPT
        + "</script><script type='application/json' id='report-data'>"
        + payload.replace("<", "\\u003c")
        + "</script></body></html>\n"
    )


@dataclass
class _OperatorView:
    batch_size: int
    latency_s: float
    bound: str
    network_share: Optional[float]
    calls: int = 0


def _network_share(operator_s: float, network_s: Optional[float]) -> Optional[float]:
    """Operator analytic time as a percent of the scheduled network latency."""
    if network_s is None or network_s <= 0:
        return None
    return operator_s / network_s * 100.0


def _oom_note(report: LatencyReport) -> str:
    entries = []
    for point in report.points:
        if not point.oom:
            continue
        memory = "可用显存 —" if point.memory_available_gb is None else f"可用显存 {point.memory_available_gb:.2f} GB"
        aria = f"batch {point.batch_size} 显存不足 · {memory}"
        entries.append(
            f"<span aria-label='{html.escape(aria, quote=True)}'><b>batch {point.batch_size}</b>{html.escape(memory)}</span>"
        )
    if not entries:
        return ""
    return (
        "<div class='oom-summary'><strong>显存不足</strong>"
        + "".join(entries)
        + "<small>图中红色背景区与圆点表示显存不足。</small></div>"
    )


def _network_batch_panel(report: LatencyReport) -> str:
    rows = [
        "<aside class='network-batch-panel'><div class='filter-head'><strong>Batch 读数</strong></div>",
        "<p class='filter-help'>点击 batch 高亮对应圆点并查看完整读数。</p>",
    ]
    for point in report.points:
        latency = "—" if point.latency_s is None else f"{point.latency_s * 1e3:.3f} ms"
        memory = "" if point.memory_available_gb is None else f" · 显存 {point.memory_available_gb:.2f} GB"
        status = "OOM" if point.oom else BOUND_LABELS.get(point.bound, point.bound)
        rows.append(
            f"<button type='button' class='network-batch-row{' oom-row' if point.oom else ''}' "
            f"data-batch='{point.batch_size}'>"
            f"<span><b>batch {point.batch_size}</b><strong>{latency}</strong></span>"
            f"<small>{html.escape(status + memory)}</small></button>"
        )
    rows.append("</aside>")
    return "".join(rows)


def _operator_series(points: Sequence[SweepPoint], name: str) -> list[_OperatorView]:
    series = []
    for point in points:
        match = next((sample for sample in point.operators if sample.name == name), None)
        if match is None:
            continue
        series.append(
            _OperatorView(
                point.batch_size,
                match.latency_s,
                match.bound,
                _network_share(match.total_s, point.latency_s),
                match.calls,
            )
        )
    return series


def _share_sentence(series: Sequence[_OperatorView]) -> str:
    present = [sample for sample in series if sample.network_share is not None]
    if not present:
        return ""
    first, last = present[0], present[-1]
    if first.batch_size == last.batch_size:
        return f" 占整网 {first.network_share:.1f}%。"
    return (
        f" 占整网从 batch {first.batch_size} 的 {first.network_share:.1f}%"
        f" 到 batch {last.batch_size} 的 {last.network_share:.1f}%。"
    )


def _download_name(report: LatencyReport) -> str:
    stem = report.model_id.replace("\\", "/").rstrip("/").split("/")[-1] or "latency-trend"
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in stem).strip("-")
    phase = report.phase if report.phase in {"decode", "prefill"} else "report"
    return f"{cleaned or 'latency-trend'}-{phase}.html"


def _meta_line(report: LatencyReport) -> str:
    return (
        f"{report.model_id} · {report.device} · {report.phase} · "
        f"query={report.query_length} context={report.context_length} · "
        f"TP={report.tp_size} · quant={report.quantize_linear} · {report.performance_model}"
    )


def _display_model_name(model_id: str) -> str:
    stem = model_id.replace("\\", "/").rstrip("/").split("/")[-1] or "Model"
    parts = []
    for part in stem.replace("-", "_").split("_"):
        if not part:
            continue
        parts.append(part.upper() if part[:-1].isdigit() and part[-1:].lower() == "b" else part.capitalize())
    return " ".join(parts)


def _meta_badges(report: LatencyReport) -> str:
    items = (
        report.device,
        f"Q {report.query_length} · Context {report.context_length}",
        f"TP {report.tp_size}",
        report.quantize_linear,
        report.performance_model,
    )
    return "<div class='meta-badges'>" + "".join(
        f"<span>{html.escape(str(item))}</span>" for item in items
    ) + "</div>"


def _share_from_percent(percent: Mapping[str, float], scale: float) -> BoundShare:
    weight = scale if scale > 0 else 1.0

    def _part(key: str) -> float:
        return float(percent.get(key, 0.0) or 0.0) / 100.0 * weight

    return BoundShare(
        memory_s=_part(MEMORY_BOUND),
        communication_s=_part(COMMUNICATION_BOUND),
        mma_s=_part(COMPUTE_BOUND_MMA),
        gp_s=_part(COMPUTE_BOUND_GP),
    )


def report_from_payload(payload: Mapping[str, Any]) -> LatencyReport:
    """Rebuild a report from the JSON written next to the HTML file."""
    points = []
    for raw_point in payload["points"]:
        operators = []
        for raw_operator in raw_point.get("operators", []):
            latency_s = float(raw_operator["latency_ms"]) / 1e3
            operators.append(
                OperatorSample(
                    name=raw_operator["name"],
                    latency_s=latency_s,
                    calls=int(raw_operator.get("calls", 0)),
                    share=_share_from_percent(raw_operator.get("components_percent", {}), latency_s),
                )
            )
        raw_latency = raw_point.get("latency_ms")
        latency_s = None if raw_latency is None else float(raw_latency) / 1e3
        scale = latency_s if latency_s is not None else 0.0
        raw_available = raw_point.get("memory_available_gb")
        points.append(
            SweepPoint(
                batch_size=int(raw_point["batch_size"]),
                latency_s=latency_s,
                classified=_share_from_percent(raw_point.get("classified_percent", {}), scale),
                components=_share_from_percent(raw_point.get("components_percent", {}), scale),
                operators=tuple(operators),
                memory_available_gb=None if raw_available is None else float(raw_available),
                oom=bool(raw_point.get("oom", False)),
            )
        )
    return LatencyReport(
        model_id=payload["model_id"],
        device=payload["device"],
        phase=payload["phase"],
        query_length=int(payload["query_length"]),
        context_length=int(payload["context_length"]),
        tp_size=int(payload["tp_size"]),
        quantize_linear=payload["quantize_linear"],
        performance_model=payload["performance_model"],
        points=points,
        top_k=len(payload.get("top_operators") or []) or 10,
        command=str(payload.get("command") or ""),
    )


def _command_block(command: str) -> str:
    """Show the exact text-generate command at the top of the report."""
    text = command.strip()
    if not text:
        return ""
    return (
        "<section class='command'>"
        "<div class='command-bar'><span>命令</span>"
        "<button type='button' id='copy-command'>复制</button></div>"
        "<pre id='command-text'>" + html.escape(text) + "</pre>"
        "</section>"
    )


def _operator_tags(report: LatencyReport, names: Sequence[str]) -> str:
    labels = _operator_labels(names)
    buttons = [
        "<div class='filter-head'><strong>算子筛选</strong>"
        "<button type='button' class='filter-reset tag on' data-op='*'>全部恢复</button></div>",
        "<p class='filter-help'><b>#</b> 表示 analytic total 排名；<b>shape N</b> 表示同名算子的第 N 个输入 shape 分组。点击一项只看该曲线，纵轴会自动缩放。</p>",
    ]
    for index, name in enumerate(names):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        series = _operator_series(report.points, name)
        bound = series[0].bound if series else UNKNOWN_BOUND
        shape = name.split(" ", 1)[1] if " " in name else "无 shape 信息"
        buttons.append(
            "<div class='filter-row-wrap'>"
            f"<button type='button' class='tag filter-row' data-op='{html.escape(name, quote=True)}' "
            f"title='{html.escape(name, quote=True)}'>"
            f"<i style='background:{color}'></i><span><b>{html.escape(labels[name])}</b>"
            f"<small>覆盖 {len(series)} 个 batch · {html.escape(BOUND_LABELS.get(bound, bound))}</small>"
            f"<em>{html.escape(shape)}</em>"
            "</span></button>"
            "<button type='button' class='full-name-toggle' aria-expanded='false'>完整名称</button>"
            f"<div class='full-name-detail' hidden><code>{html.escape(name)}</code></div>"
            "</div>"
        )
    return "<aside class='operator-filter' id='op-tags'>" + "".join(buttons) + "</aside>"


def _operator_detail_panels(
    report: LatencyReport, names: Sequence[str], labels: Mapping[str, str]
) -> str:
    panels = [
        "<section class='operator-details' id='op-details'>",
        "<div class='detail-heading'><h3>算子详情</h3><span>点击上方 label 查看 shape 与占整网时间</span></div>",
        "<p class='detail-empty'>请选择一个算子；“全部恢复”用于回到完整对比。</p>",
    ]
    for name in names:
        op_name, _, shape = name.partition(" ")
        rows = []
        total_calls = 0
        for point in report.points:
            sample = next((item for item in point.operators if item.name == name), None)
            if sample is None:
                continue
            total_calls += sample.calls
            total_ms = sample.total_s * 1e3
            network_ms = None if point.latency_s is None else point.latency_s * 1e3
            share = _network_share(sample.total_s, point.latency_s)
            rows.append(
                "<tr>"
                f"<td>{point.batch_size}</td><td>{sample.latency_s * 1e3:.4f}</td>"
                f"<td>{sample.calls}</td><td>{total_ms:.4f}</td>"
                f"<td>{'—' if network_ms is None else f'{network_ms:.4f}'}</td>"
                f"<td>{'—' if share is None else f'{share:.1f}%'}</td>"
                f"<td>{html.escape(BOUND_LABELS.get(sample.bound, sample.bound))}</td>"
                "</tr>"
            )
        panels.append(
            f"<article class='op-detail-panel' data-op='{html.escape(name, quote=True)}' hidden>"
            f"<div class='detail-title'><strong>{html.escape(labels[name])}</strong>"
            f"<span>覆盖 {len(rows)} 个 batch · 该组共调用 {total_calls} 次</span></div>"
            f"<dl><dt>算子名</dt><dd><code>{html.escape(op_name)}</code></dd>"
            f"<dt>输入 shape</dt><dd><code>{html.escape(shape or '无 shape 信息')}</code></dd></dl>"
            "<table class='detail-table'><thead><tr><th>batch</th><th>analytic avg ms</th>"
            "<th>调用次数</th><th>analytic total ms</th><th>整网时间 ms</th>"
            "<th>占整网</th><th>bound</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></article>"
        )
    panels.append("</section>")
    return "".join(panels)


def _operator_labels(names: Sequence[str]) -> dict[str, str]:
    """Ranked, unique labels; full shape stays in details/readout."""
    bases = []
    for name in names:
        parts = name.split(" [", 1)[0].split(".")
        if parts and parts[-1] in {"default", "Tensor"} and len(parts) > 1:
            parts.pop()
        bases.append(parts[-1] if parts else name)
    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    for rank, (name, base) in enumerate(zip(names, bases), start=1):
        counts[base] = counts.get(base, 0) + 1
        suffix = f" · shape {counts[base]}" if bases.count(base) > 1 else ""
        labels[name] = f"#{rank:02d} {base}{suffix}"
    return labels


def _point_mark(x: float, y: float, radius: float, fill: str, tip: str, stroke: str = "") -> str:
    """Clickable chart point. The value is stored for the readout, not only a hover title."""
    stroke_attr = f" stroke='{stroke}' stroke-width='1.5'" if stroke else ""
    return (
        f"<g class='point' tabindex='0' data-tip='{html.escape(tip, quote=True)}'>"
        f"<circle class='hit' cx='{x:.1f}' cy='{y:.1f}' r='12'/>"
        f"<circle class='dot' cx='{x:.1f}' cy='{y:.1f}' r='{radius}' fill='{fill}'{stroke_attr}>"
        f"<title>{html.escape(tip)}</title></circle></g>"
    )


def _legend() -> str:
    items = "".join(
        f"<span><i style='background:{BOUND_COLORS[key]}'></i>{html.escape(BOUND_LABELS[key])}</span>"
        for key in BOUND_ORDER
        if key != UNKNOWN_BOUND
    )
    return f"<div class='legend'>{items}</div>"


def _latency_chart(
    title: str,
    batches: Sequence[int],
    values_ms: Sequence[Optional[float]],
    bounds: Sequence[str],
    *,
    width: int,
    height: int,
    shares: Optional[Sequence[Optional[float]]] = None,
    oom: Optional[Sequence[bool]] = None,
    available_gb: Optional[Sequence[Optional[float]]] = None,
    metric: str = "",
) -> str:
    if not batches:
        return "<p class='note'>没有数据。</p>"
    pad_l, pad_r, pad_t, pad_b = 58, 58, 22, 36
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    finite = [value for value in values_ms if value is not None]
    y_min = 0.0
    y_max = max(finite) if finite else 1.0
    if y_max <= 0:
        y_max = 1.0
    y_max *= 1.12

    def x_at(index: int) -> float:
        if len(batches) == 1:
            return pad_l + plot_w / 2
        return pad_l + plot_w * index / (len(batches) - 1)

    def y_at(value: float) -> float:
        return pad_t + plot_h * (1 - (value - y_min) / (y_max - y_min))

    bands = []
    for index, bound in enumerate(bounds):
        left = pad_l if index == 0 else (x_at(index - 1) + x_at(index)) / 2
        right = width - pad_r if index == len(bounds) - 1 else (x_at(index) + x_at(index + 1)) / 2
        color = BOUND_COLORS.get(bound, BOUND_COLORS[UNKNOWN_BOUND])
        bands.append(
            f"<rect x='{left:.1f}' y='{pad_t}' width='{max(right - left, 0):.1f}' height='{plot_h}' "
            f"fill='{color}' fill-opacity='0.16'></rect>"
        )
    oom_layer: list[str] = []
    oom_indexes = [index for index, is_oom in enumerate(oom or ()) if is_oom and index < len(batches)]
    if oom_indexes:
        first_oom = oom_indexes[0]
        boundary_x = x_at(first_oom)
        oom_layer.extend(
            [
                f"<rect class='oom-region' x='{boundary_x:.1f}' y='{pad_t}' width='{max(width - pad_r - boundary_x, 0):.1f}' "
                f"height='{plot_h}' fill='#b42318' fill-opacity='0.075'/>",
                f"<line class='oom-boundary' x1='{boundary_x:.1f}' y1='{pad_t}' x2='{boundary_x:.1f}' y2='{pad_t + plot_h}' "
                "stroke='#b42318' stroke-width='1.5' stroke-dasharray='5 4'/>",
            ]
        )
    coords = []
    dots = []
    for index, (batch, value, bound) in enumerate(zip(batches, values_ms, bounds)):
        is_oom = bool(oom and index < len(oom) and oom[index])
        if value is None:
            continue
        x = x_at(index)
        y = y_at(value)
        coords.append(f"{x:.1f},{y:.1f}")
        share = None if shares is None or index >= len(shares) else shares[index]
        share_text = "" if share is None else f" · 占整网 {share:.1f}%"
        available = None if available_gb is None or index >= len(available_gb) else available_gb[index]
        oom_text = ""
        if is_oom:
            oom_text = " · OOM"
            if available is not None:
                oom_text += f" · 可用显存 {available:.2f} GB"
        label = (
            f"{title} · batch {batch} · {metric + ' ' if metric else ''}{value:.3f} ms · "
            f"{BOUND_LABELS.get(bound, bound)}{share_text}{oom_text}"
        )
        dots.append(
            _point_mark(
                x,
                y,
                4.5,
                "#b42318" if is_oom else BOUND_COLORS.get(bound, BOUND_COLORS[UNKNOWN_BOUND]),
                label,
            )
        )
        dots[-1] = dots[-1].replace(
            "class='point'",
            f"class='point network-point{' oom-point' if is_oom else ''}' data-batch='{batch}'",
            1,
        )
        if is_oom and oom_indexes and index == oom_indexes[0]:
            label_x = min(x + 8, width - pad_r - 28)
            label_y = max(y - 10, pad_t + 13)
            dots.append(
                f"<text x='{label_x:.1f}' y='{label_y:.1f}' class='oom first-oom-label'>OOM</text>"
            )
    y_ticks = []
    for step in range(4):
        value = y_max * step / 3
        y = y_at(value)
        y_ticks.append(
            f"<line x1='{pad_l}' y1='{y:.1f}' x2='{width - pad_r}' y2='{y:.1f}' stroke='#e4ddd2'/>"
            f"<text x='{pad_l - 8}' y='{y + 4:.1f}' text-anchor='end' class='tick'>{value:.3g}</text>"
        )
    x_labels = []
    for index, batch in enumerate(batches):
        x_labels.append(
            f"<text x='{x_at(index):.1f}' y='{height - 12}' text-anchor='middle' class='tick'>{batch}</text>"
        )
    changes = bound_transitions(
        list(zip(batches, bounds)),
        lambda item: item[1],
        lambda item: item[0],
    )
    markers = []
    for change in changes:
        index = batches.index(change.batch_size)
        x = x_at(index)
        markers.append(
            f"<line x1='{x:.1f}' y1='{pad_t}' x2='{x:.1f}' y2='{pad_t + plot_h}' "
            f"stroke='#1c1915' stroke-dasharray='3 3'/>"
            f"<text x='{x + 4:.1f}' y='{pad_t + 14}' class='marker'>"
            f"{html.escape(BOUND_LABELS.get(change.previous, change.previous))} → "
            f"{html.escape(BOUND_LABELS.get(change.current, change.current))}</text>"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)} latency'>"
        + "".join(bands)
        + "".join(oom_layer)
        + "".join(y_ticks)
        + "".join(x_labels)
        + (f"<polyline fill='none' stroke='#1c1915' stroke-width='2' points='{' '.join(coords)}'/>" if coords else "")
        + "".join(dots)
        + "".join(markers)
        + f"<text x='{pad_l}' y='12' class='axis'>ms</text>"
        + "</svg>"
    )


def _series_values(
    report: LatencyReport,
    name: str,
) -> tuple[list[Optional[float]], list[str], list[Optional[float]]]:
    values: list[Optional[float]] = []
    bounds: list[str] = []
    shares: list[Optional[float]] = []
    for point in report.points:
        match = next((sample for sample in point.operators if sample.name == name), None)
        if match is None:
            values.append(None)
            bounds.append(UNKNOWN_BOUND)
            shares.append(None)
        else:
            values.append(match.latency_s * 1e3)
            bounds.append(match.bound)
            shares.append(_network_share(match.total_s, point.latency_s))
    return values, bounds, shares


def _top_trend_chart(report: LatencyReport) -> str:
    """One shared-axis chart of the top operators' latency versus batch size."""
    names = report.top_operators()
    batches = [point.batch_size for point in report.points]
    if not names or not batches:
        return "<p class='note'>没有算子数据。</p>"
    width, height = 960, 420
    pad_l, pad_r, pad_t, pad_b = 58, 150, 18, 36
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    prepared = [(name, *_series_values(report, name)) for name in names]
    labels = _operator_labels(names)
    finite = [value for _name, values, _bounds, _shares in prepared for value in values if value is not None]
    y_max = max(finite) * 1.12 if finite else 1.0
    if y_max <= 0:
        y_max = 1.0

    def x_at(index: int) -> float:
        if len(batches) == 1:
            return pad_l + plot_w / 2
        return pad_l + plot_w * index / (len(batches) - 1)

    def y_at(value: float) -> float:
        return pad_t + plot_h * (1 - value / y_max)

    parts = []
    for step in range(4):
        value = y_max * step / 3
        y = y_at(value)
        parts.append(
            f"<line x1='{pad_l}' y1='{y:.1f}' x2='{width - pad_r}' y2='{y:.1f}' stroke='#e4ddd2'/>"
            f"<text x='{pad_l - 8}' y='{y + 4:.1f}' text-anchor='end' class='tick y-tick' "
            f"data-step='{step}'>{value:.3g}</text>"
        )
    for index, batch in enumerate(batches):
        parts.append(
            f"<text x='{x_at(index):.1f}' y='{height - 12}' text-anchor='middle' class='tick'>{batch}</text>"
        )
    for series_index, (name, values, bounds, shares) in enumerate(prepared):
        color = SERIES_COLORS[series_index % len(SERIES_COLORS)]
        all_coords: list[str] = []
        dots: list[str] = []
        previous = None
        for index, value in enumerate(values):
            if value is None:
                previous = None
                continue
            x = x_at(index)
            y = y_at(value)
            all_coords.append(f"{x:.1f},{y:.1f}")
            bound = bounds[index]
            change = ""
            if previous is not None and previous != bound:
                change = f" · {BOUND_LABELS.get(previous, previous)} → {BOUND_LABELS.get(bound, bound)}"
            previous = bound
            share = shares[index]
            share_text = "" if share is None else f" · 占整网 {share:.1f}%"
            label = (
                f"{name} · batch {batches[index]} · analytic avg {value:.3f} ms · "
                f"{BOUND_LABELS.get(bound, bound)}{change}{share_text}"
            )
            radius = 5 if change else 3.5
            dots.append(
                _point_mark(
                    x,
                    y,
                    radius,
                    BOUND_COLORS.get(bound, color),
                    label,
                    stroke=color,
                )
            )
            dots[-1] = dots[-1].replace(
                "class='point'",
                f"class='point' data-x='{x:.1f}' data-value='{value:.8f}' data-base-y='{y:.1f}'",
                1,
            )
        if len(all_coords) > 1:
            dots.insert(0, f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(all_coords)}'/>")
        present = [(index, value) for index, value in enumerate(values) if value is not None]
        if present:
            last_index, last_value = present[-1]
            label_x = x_at(last_index) + 8
            label_y = y_at(last_value) + (series_index % 3 - 1) * 12
            dots.append(
                f"<text class='series-label' x='{label_x:.1f}' y='{label_y:.1f}' fill='{color}' "
                f"data-value='{last_value:.8f}' data-base-y='{label_y:.1f}'>"
                f"{html.escape(labels[name])}</text>"
            )
        parts.append(
            f"<g class='op-series' data-op='{html.escape(name, quote=True)}' "
            f"data-label='{html.escape(labels[name], quote=True)}'>" + "".join(dots) + "</g>"
        )
    parts.append(f"<text x='{pad_l}' y='12' class='axis'>ms</text>")
    return (
        "<div id='op-compare'><svg viewBox='0 0 "
        + f"{width} {height}' role='img' data-plot-top='{pad_t}' data-plot-height='{plot_h}' "
        + f"data-y-max='{y_max:.8f}' role='img' aria-label='top operator latency trends'>"
        + "".join(parts)
        + "</svg></div>"
    )


def _top_latency_table(report: LatencyReport) -> str:
    names = report.top_operators()
    labels = _operator_labels(names)
    batches = [point.batch_size for point in report.points]
    header = "<th>算子</th>" + "".join(f"<th>{batch}</th>" for batch in batches) + "<th>bound 变化</th>"
    rows = []
    for name in names:
        values, bounds, shares = _series_values(report, name)
        cells = []
        for value, share in zip(values, shares):
            if value is None:
                cells.append("<td>—</td>")
                continue
            share_text = "—" if share is None else f"{share:.1f}%"
            cells.append(f"<td>{value:.3f}<div class='share'>{share_text}</div></td>")
        changes = bound_transitions(
            [item for item in zip(batches, bounds, values) if item[2] is not None],
            lambda item: item[1],
            lambda item: item[0],
        )
        change_text = "；".join(
            f"batch {change.batch_size}: {BOUND_LABELS.get(change.previous, change.previous)} → "
            f"{BOUND_LABELS.get(change.current, change.current)}"
            for change in changes
        )
        rows.append(
            "<tr class='op-row' data-op='"
            + html.escape(name, quote=True)
            + "'><td class='op'>"
            + "<strong>" + html.escape(labels[name]) + "</strong>"
            + "<details><summary>完整名称</summary><code>" + html.escape(name) + "</code></details>"
            + "</td>"
            + "".join(cells)
            + "<td>"
            + html.escape(change_text)
            + "</td></tr>"
        )
    return (
        "<table class='ops'><caption>上面是同一 shape 的 analytic avg（毫秒），"
        "下面是这一组 analytic total 占整网调度时间的比例。</caption><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _mix_chart(batches: Sequence[int], shares: Sequence[BoundShare]) -> str:
    if not batches:
        return ""
    width = 960
    row_h, pad_l, pad_r, pad_t, pad_b = 28, 64, 20, 8, 8
    height = pad_t + pad_b + row_h * len(batches)
    bar_x = pad_l
    bar_w = width - pad_l - pad_r
    parts = []
    for index, (batch, share) in enumerate(zip(batches, shares)):
        percents = share.percent()
        y = pad_t + index * row_h + 5
        parts.append(
            f"<text x='{pad_l - 10}' y='{y + 12:.1f}' text-anchor='end' class='tick'>batch {batch}</text>"
            f"<rect x='{bar_x}' y='{y}' width='{bar_w}' height='16' rx='4' fill='#eef0f4'/>"
        )
        x = bar_x
        for key in (MEMORY_BOUND, COMMUNICATION_BOUND, COMPUTE_BOUND_MMA, COMPUTE_BOUND_GP):
            percent = percents.get(key, 0.0)
            segment_w = bar_w * percent / 100.0
            if segment_w <= 0:
                continue
            parts.append(
                f"<rect x='{x:.1f}' y='{y}' width='{segment_w:.1f}' height='16' fill='{BOUND_COLORS[key]}'>"
                f"<title>batch {batch} · {html.escape(BOUND_LABELS[key])} {percent:.1f}%</title></rect>"
            )
            if percent >= 12:
                parts.append(
                    f"<text x='{x + segment_w / 2:.1f}' y='{y + 12:.1f}' text-anchor='middle' "
                    f"class='bar-label'>{percent:.1f}%</text>"
                )
            x += segment_w
    return f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='bound mix'>" + "".join(parts) + "</svg>"


def _network_table(report: LatencyReport) -> str:
    rows = []
    previous = None
    for point in report.points:
        percents = point.classified.percent()
        changed = "" if previous is None or previous == point.bound else "切换"
        previous = point.bound
        latency_cell = "—" if point.latency_s is None else f"{point.latency_s * 1e3:.4f}"
        bound_text = BOUND_LABELS.get(point.bound, point.bound)
        status = "OOM" if point.oom else "正常"
        memory = "—" if point.memory_available_gb is None else f"{point.memory_available_gb:.2f}"
        rows.append(
            "<tr>"
            f"<td>{point.batch_size}</td>"
            f"<td>{latency_cell}</td>"
            f"<td>{html.escape(bound_text)}</td>"
            f"<td>{percents[MEMORY_BOUND]:.1f}%</td>"
            f"<td>{percents[COMMUNICATION_BOUND]:.1f}%</td>"
            f"<td>{percents[COMPUTE_BOUND_MMA]:.1f}%</td>"
            f"<td>{percents[COMPUTE_BOUND_GP]:.1f}%</td>"
            f"<td>{memory}</td><td class='status {'oom-cell' if point.oom else ''}'>{status}</td>"
            f"<td>{changed or '—'}</td>"
            "</tr>"
        )
    return (
        "<section class='operator-details network-details'>"
        "<div class='detail-heading'><h3>整网详情</h3>"
        "<span>调度后的设备执行时间与 BoundAnalyzer 构成</span></div>"
        "<div class='network-table-wrap'><table class='detail-table network-result-table'>"
        "<thead><tr><th>batch</th><th>整网时间 ms</th><th>主导 bound</th>"
        "<th>访存</th><th>通信</th><th>矩阵计算</th><th>通用计算</th>"
        "<th>可用显存 GB</th><th>状态</th><th>bound 变化</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></section>"
    )


_STYLE = """
body{margin:0;background:#f4f5f7;color:#1f2430;
font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI","PingFang SC","Noto Sans SC",sans-serif}
body>*{max-width:1360px}
header.page,p,section{margin-left:auto;margin-right:auto}
header.page{background:#fff;border:1px solid #e6e8ee;border-radius:16px;
margin:20px auto 0;padding:20px 22px 16px}
.page-top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px}
.page-top>div{min-width:0}
h1{font-size:22px;font-weight:650;letter-spacing:-.02em;margin:0 0 4px}
.toolbar{margin:0}
button{border:1px solid #e6e8ee;background:#fff;color:#1f2430;border-radius:8px;
padding:6px 10px;font:13px/1.3 ui-sans-serif,system-ui,sans-serif;cursor:pointer;white-space:nowrap}
button:hover{border-color:#c5cad3}
h2{font-size:16px;font-weight:650;margin:14px 0 6px}
h3{font-size:13px;font-weight:650;margin:12px 0 4px;color:#4b5160}
.meta,.note{color:#6b7280;margin:0}.header-kicker{margin:1px 0 0;color:#6b7280;font-size:12px}
.meta-badges{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}.meta-badges span{display:inline-flex;align-items:center;
padding:3px 8px;border:1px solid #e3e6ec;background:#f7f8fa;border-radius:999px;color:#4b5563;font-size:11px}
.command{margin-top:14px;padding-top:12px;border-top:1px solid #eef0f4}
.command-bar{display:flex;align-items:center;justify-content:space-between;gap:8px;color:#6b7280;font-size:12px}
#command-text{margin:8px 0 0;padding:12px 14px;background:#f7f8fa;color:#1f2430;border:1px solid #e6e8ee;
border-radius:10px;overflow:auto;white-space:pre;font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
section{background:#fff;border:1px solid #e6e8ee;border-radius:16px;margin:14px auto;padding:6px 18px 16px}
header.page section.command{background:transparent;border:0;border-radius:0;margin:14px 0 0;
padding:12px 0 0;border-top:1px solid #eef0f4}
.section-tools{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:10px 0 4px;padding:7px 9px;
background:#f7f8fa;border:1px solid #e7e9ee;border-radius:9px}.section-readout{margin:0!important;color:#4b5563;font-size:11px}
.section-tools .legend{padding:0;justify-content:flex-end;flex:0 0 auto}
.point{cursor:pointer}
.point .hit{fill:transparent}
.point.on .dot{stroke:#111827;stroke-width:2.5}
.share{color:#6b7280;font:11px/1.3 ui-sans-serif,system-ui,sans-serif}
caption{caption-side:bottom;text-align:left;color:#6b7280;
font:12px/1.4 ui-sans-serif,system-ui,sans-serif;padding-top:.3rem}
.legend{display:flex;flex-wrap:wrap;gap:8px 14px;padding:4px 2px 0}
.view-switch{display:flex;align-items:center;justify-content:flex-end;gap:8px;margin:8px 0 10px}.view-mode{display:flex}
.view-switch .view-button{border-radius:0;margin-left:-1px}.view-switch .view-button:first-child{border-radius:8px 0 0 8px}
.view-switch .view-button:last-child{border-radius:0 8px 8px 0}.view-button.on{background:#111827;color:#fff}
.op-analysis{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:12px;align-items:start}
#op-compare{min-width:0;border:1px solid #e6e8ee;border-radius:12px;padding:8px;background:#fff}
.operator-filter{border:1px solid #d8dde6;border-radius:12px;background:#f8f9fb;padding:10px;max-height:430px;overflow:auto}
.filter-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:2px}
#toggle-all-names{padding:4px 7px;border:1px solid #d4d9e2;border-radius:6px;background:#fff;color:#4b5563;font-size:11px}
#toggle-all-names:hover{border-color:#9ca3af;color:#111827}
.filter-reset{font-size:11px!important;padding:4px 7px!important;width:auto!important}.filter-help{margin:0 0 8px;color:#6b7280;font-size:11px}
.filter-row-wrap{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;border-radius:8px}
.filter-row-wrap:hover{background:#eef1f5}.filter-row-wrap .tag.filter-row:hover{background:transparent}
.full-name-toggle{margin-right:5px;padding:3px 6px;border:1px solid #d4d9e2;border-radius:6px;background:#fff;color:#4b5563;font-size:10px;white-space:nowrap}
.full-name-toggle:hover{border-color:#9ca3af;color:#111827}.full-name-detail{grid-column:1/-1;margin:0 6px 7px;padding:7px 8px;border-radius:7px;background:#fff;border:1px solid #e1e5eb}
.full-name-detail code{display:block;overflow-wrap:anywhere;color:#374151;font:10px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.tag.filter-row{display:flex;width:100%;align-items:center;gap:8px;text-align:left;border:0;border-radius:8px;background:transparent;
padding:7px 6px;color:#374151}.tag.filter-row:hover{background:#eef1f5}.tag.filter-row.on{background:#e5e7eb;color:#111827}
.tag.filter-row i{width:10px;height:10px;flex:0 0 10px;border-radius:50%}.tag.filter-row span{min-width:0;display:block}
.tag.filter-row b{display:block;overflow:hidden;text-overflow:ellipsis}.tag.filter-row small{display:block;color:#7b8190;font-size:10px}
.tag.filter-row em{display:block;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
color:#4b5563;font:10px/1.35 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.series-label{font:600 10px/1 ui-sans-serif,system-ui,sans-serif;paint-order:stroke;stroke:#fff;stroke-width:4px;stroke-linejoin:round}
.operator-details{max-width:none!important;margin:14px 0 0!important;padding:0!important;border:1px solid #dfe3ea!important;
border-radius:12px!important;overflow:hidden;background:#fff!important}
.detail-heading{display:flex;align-items:baseline;gap:10px;padding:10px 14px;background:#f6f7f9;border-bottom:1px solid #e5e7eb}
.detail-heading h3{margin:0;color:#111827;font-size:14px}.detail-heading span{color:#6b7280;font-size:11px}
.detail-empty{margin:0!important;padding:18px 14px;color:#6b7280}.op-detail-panel{padding:14px}
.detail-title{display:flex;align-items:baseline;gap:10px;margin-bottom:10px}.detail-title strong{font-size:16px}
.detail-title span{color:#6b7280;font-size:12px}.op-detail-panel dl{display:grid;grid-template-columns:70px minmax(0,1fr);gap:6px 10px;margin:0 0 12px}
.op-detail-panel dt{color:#6b7280;font-size:12px}.op-detail-panel dd{margin:0;min-width:0}.op-detail-panel dd code{display:block;
padding:6px 8px;background:#f5f6f8;border-radius:6px;overflow-wrap:anywhere;font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.detail-table th,.detail-table td{text-align:right!important;white-space:nowrap}.detail-table th:first-child,.detail-table td:first-child{text-align:left!important}
.network-table-wrap{padding:0 14px 12px;overflow-x:auto}.network-result-table{margin-top:10px}
.network-result-table .status{font-weight:650}.network-result-table .oom-cell{color:#b42318}
.grid.solo{grid-template-columns:1fr}
[hidden]{display:none !important}
table.ops td.op,table.ops th:first-child{text-align:left}
.legend span{display:inline-flex;align-items:center;gap:6px;color:#4b5160;font-size:12px}
.legend i{width:8px;height:8px;border-radius:99px;display:inline-block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.card{background:#fafbfc;border:1px solid #e6e8ee;border-radius:12px;padding:8px 10px 2px}
.card h3{font-size:15px;color:#1f2430}.coverage{margin:4px 0;color:#6b7280;font-size:12px}
details{margin:3px 0;color:#6b7280;font-size:12px}summary{cursor:pointer;white-space:nowrap}
details code{display:block;margin-top:5px;padding:7px 8px;background:#f1f3f6;border-radius:7px;
overflow-wrap:anywhere;color:#374151;font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
svg{width:100%;height:auto;display:block}
.tick,.axis{font:11px/1 ui-sans-serif,system-ui,sans-serif;fill:#6b7280}
.marker{font:11px/1 ui-sans-serif,system-ui,sans-serif;fill:#1f2430}
.oom{font:11px/1 ui-sans-serif,system-ui,sans-serif;fill:#b42318;font-weight:650}
.oom-badge rect{fill:#fff7f6;stroke:#efb0aa}.oom-badge line{stroke:#b42318;stroke-width:1;stroke-dasharray:2 2}
.oom-memory{fill:#8f2d24;font:8px/1 ui-sans-serif,system-ui,sans-serif}.bar-label{fill:#fff;font:600 9px/1 ui-sans-serif,system-ui,sans-serif}
.oom-summary{display:flex;align-items:center;flex-wrap:wrap;gap:7px;margin:8px 0 12px;padding:8px 10px;
border:1px solid #f0c7c3;background:#fff8f7;border-radius:9px;color:#8f2d24;font-size:11px}
.oom-summary>strong{font-size:12px}.oom-summary span{display:inline-flex;gap:5px;padding:3px 7px;background:#fff;border:1px solid #f1d6d3;border-radius:999px}
.oom-summary small{color:#9a5b55}
.network-analysis{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:12px;align-items:start;margin-top:10px}
.network-analysis>svg{box-sizing:border-box;border:1px solid #e6e8ee;border-radius:12px;padding:8px;background:#fff}
.network-batch-panel{border:1px solid #d8dde6;border-radius:12px;background:#f8f9fb;padding:10px;max-height:360px;overflow:auto}
.network-batch-panel .filter-head{position:sticky;top:-10px;z-index:2;margin:-10px -10px 0;padding:10px 10px 5px;background:#f8f9fb}
.network-batch-row{display:block;width:100%;border:0;background:transparent;text-align:left;padding:7px 7px;border-radius:8px}
.network-batch-row:hover,.network-batch-row.on{background:#e5e7eb}.network-batch-row span{display:flex;align-items:center;justify-content:space-between;gap:8px}
.network-batch-row b{font-size:12px}.network-batch-row strong{font-size:11px;font-variant-numeric:tabular-nums}
.network-batch-row small{display:block;color:#6b7280;font-size:10px}.network-batch-row.oom-row small{color:#b42318}
table{border-collapse:collapse;width:100%;background:#fff;margin:8px 0 4px;font-size:13px}
th,td{text-align:right;padding:8px 10px;border-bottom:1px solid #eef0f4;font-variant-numeric:tabular-nums}
th:nth-child(3),td:nth-child(3),th:last-child,td:last-child{text-align:left}
th{font-size:12px;color:#6b7280;font-weight:650}
tr:hover td{background:#fafbfc}
@media(max-width:1000px){
.op-analysis{grid-template-columns:1fr}.operator-filter{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
gap:2px 10px;max-height:300px}.filter-head,.filter-help{grid-column:1/-1}.tag.filter-row{min-width:0}
.network-analysis{grid-template-columns:1fr}.network-batch-panel{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
gap:2px 10px;max-height:260px}.network-batch-panel .filter-head,.network-batch-panel .filter-help{grid-column:1/-1}
}
@media(max-width:720px){
body>*{max-width:calc(100% - 16px)}header.page{margin-top:8px;padding:14px}
section{padding:4px 12px 12px;margin-top:8px}.page-top{align-items:center}.meta{font-size:12px}
h1{font-size:18px}.toolbar{flex:0 0 auto}.command-bar{flex-wrap:nowrap}
#command-text{font-size:11px}.operator-filter{grid-template-columns:1fr;max-height:320px}
.network-batch-panel{grid-template-columns:1fr;max-height:300px}
.view-switch{margin:8px 0}.grid{grid-template-columns:1fr}table{display:block;overflow-x:auto;white-space:nowrap}
.section-tools{align-items:flex-start;flex-direction:column}.section-tools .legend{width:100%;justify-content:flex-start;flex-wrap:nowrap;overflow-x:auto}
}
"""

_FILTER_SCRIPT = """
const tags=document.querySelectorAll('#op-tags .tag');
function selectOp(name){
  const all=name==='*';
  document.querySelectorAll('.op-series,.op-row').forEach(el=>{
    el.toggleAttribute('hidden', !all && el.dataset.op!==name);
  });
  document.querySelectorAll('.op-detail-panel').forEach(el=>{
    el.toggleAttribute('hidden', all || el.dataset.op!==name);
  });
  document.querySelector('.detail-empty')?.toggleAttribute('hidden',!all);
  const compare=document.getElementById('op-compare');
  if(compare) compare.hidden=false;
  tags.forEach(btn=>btn.classList.toggle('on', btn.dataset.op===name));
  rescaleCompare(name);
}
function rescaleCompare(name){
  const svg=document.querySelector('#op-compare svg');
  if(!svg) return;
  const top=Number(svg.dataset.plotTop), height=Number(svg.dataset.plotHeight);
  const baseMax=Number(svg.dataset.yMax);
  const chosen=name==='*' ? [...svg.querySelectorAll('.op-series')] :
    [...svg.querySelectorAll('.op-series')].filter(el=>el.dataset.op===name);
  const values=chosen.flatMap(series=>[...series.querySelectorAll('.point')]
    .map(point=>Number(point.dataset.value)).filter(Number.isFinite));
  const max=(values.length ? Math.max(...values)*1.12 : baseMax) || 1;
  svg.querySelectorAll('.op-series').forEach(series=>{
    const pointCoords=[];
    series.querySelectorAll('.point').forEach(point=>{
      const value=Number(point.dataset.value);
      const baseY=Number(point.dataset.baseY);
      const nextY=top+height*(1-value/max);
      point.setAttribute('transform',`translate(0 ${nextY-baseY})`);
      pointCoords.push(`${Number(point.dataset.x)},${nextY}`);
    });
    series.querySelectorAll('.series-label').forEach(label=>{
      const value=Number(label.dataset.value);
      const baseY=Number(label.dataset.baseY);
      const nextY=top+height*(1-value/max);
      label.setAttribute('transform',`translate(0 ${nextY-baseY})`);
    });
    series.querySelectorAll('polyline').forEach((line,index)=>{
      line.toggleAttribute('hidden',index>0 || pointCoords.length<2);
      if(index===0) line.setAttribute('points',pointCoords.join(' '));
    });
  });
  svg.querySelectorAll('.y-tick').forEach(tick=>{
    tick.textContent=(max*Number(tick.dataset.step)/3).toPrecision(3).replace(/\\.?0+$/,'');
  });
}
tags.forEach(btn=>btn.addEventListener('click',()=>selectOp(btn.dataset.op)));
document.querySelectorAll('.full-name-toggle').forEach(btn=>btn.addEventListener('click',()=>{
  const detail=btn.parentElement?.querySelector('.full-name-detail');
  if(!detail) return;
  const expanded=detail.hasAttribute('hidden');
  detail.toggleAttribute('hidden',!expanded);
  btn.setAttribute('aria-expanded',String(expanded));
  btn.textContent=expanded?'收起名称':'完整名称';
}));
const toggleAllNames=document.getElementById('toggle-all-names');
if(toggleAllNames) toggleAllNames.addEventListener('click',()=>{
  const expand=toggleAllNames.getAttribute('aria-expanded')!=='true';
  document.querySelectorAll('.full-name-detail').forEach(detail=>detail.toggleAttribute('hidden',!expand));
  document.querySelectorAll('#op-table-view details').forEach(detail=>detail.open=expand);
  document.querySelectorAll('.full-name-toggle').forEach(btn=>{
    btn.setAttribute('aria-expanded',String(expand));
    btn.textContent=expand?'收起名称':'完整名称';
  });
  toggleAllNames.setAttribute('aria-expanded',String(expand));
  toggleAllNames.textContent=expand?'收起全部名称':'展开全部名称';
});
const viewButtons=document.querySelectorAll('.view-button');
viewButtons.forEach(btn=>btn.addEventListener('click',()=>{
  const table=btn.dataset.view==='table';
  document.getElementById('op-chart-view')?.toggleAttribute('hidden',table);
  document.getElementById('op-table-view')?.toggleAttribute('hidden',!table);
  viewButtons.forEach(item=>item.classList.toggle('on',item===btn));
}));
const readout=document.getElementById('point-readout');
const networkReadout=document.getElementById('network-readout');
function showPoint(el){
  document.querySelectorAll('.point.on').forEach(node=>node.classList.remove('on'));
  el.classList.add('on');
  const target=el.classList.contains('network-point') ? networkReadout : readout;
  if(target) target.textContent=el.dataset.tip || '';
  if(el.dataset.batch){
    document.querySelectorAll('.network-batch-row').forEach(row=>
      row.classList.toggle('on',row.dataset.batch===el.dataset.batch));
  }
}
document.querySelectorAll('.point').forEach(el=>{
  el.addEventListener('click',event=>{event.stopPropagation(); showPoint(el);});
  el.addEventListener('keydown',event=>{
    if(event.key==='Enter' || event.key===' '){event.preventDefault(); showPoint(el);}
  });
});
document.querySelectorAll('.network-batch-row').forEach(row=>{
  row.addEventListener('click',()=>{
    const point=document.querySelector(`.network-point[data-batch="${row.dataset.batch}"]`);
    if(point) showPoint(point);
  });
});
const download=document.getElementById('download-report');
if(download) download.addEventListener('click',()=>{
  const name=download.dataset.name || 'latency-trend.html';
  const html='<!doctype html>'+document.documentElement.outerHTML;
  const blob=new Blob([html],{type:'text/html;charset=utf-8'});
  const url=URL.createObjectURL(blob);
  const link=document.createElement('a');
  link.href=url;
  link.download=name;
  link.click();
  URL.revokeObjectURL(url);
});
const copyCommand=document.getElementById('copy-command');
const commandText=document.getElementById('command-text');
if(copyCommand && commandText) copyCommand.addEventListener('click',()=>{
  const text=commandText.textContent || '';
  const done=()=>{ copyCommand.textContent='已复制'; };
  if(navigator.clipboard && navigator.clipboard.writeText){
    navigator.clipboard.writeText(text).then(done).catch(()=>copyWithSelection(text, done));
  } else {
    copyWithSelection(text, done);
  }
});
function copyWithSelection(text, done){
  const area=document.createElement('textarea');
  area.value=text;
  document.body.appendChild(area);
  area.select();
  document.execCommand('copy');
  area.remove();
  done();
}
"""

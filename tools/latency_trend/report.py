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
    COMMUNICATION_BOUND,
    COMPUTE_BOUND_MMA,
    COMPUTE_BOUND_GP,
    UNKNOWN_BOUND,
)

BOUND_LABELS = {
    MEMORY_BOUND: "访存 memory",
    COMMUNICATION_BOUND: "通信 communication",
    COMPUTE_BOUND_MMA: "矩阵计算 compute",
    COMPUTE_BOUND_GP: "通用计算 compute",
    UNKNOWN_BOUND: "未分类",
}

BOUND_COLORS = {
    MEMORY_BOUND: "#c47b2b",
    COMMUNICATION_BOUND: "#6d5bd0",
    COMPUTE_BOUND_MMA: "#0f6e8c",
    COMPUTE_BOUND_GP: "#1d7a46",
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
        if self.total_s() <= 0:
            return UNKNOWN_BOUND
        result = PerformanceModel.Result(
            self.total_s(),
            {
                _COMPONENT_STATS["memory"]: self.memory_s,
                _COMPONENT_STATS["communication"]: self.communication_s,
                _COMPONENT_STATS["mma"]: self.mma_s,
                _COMPONENT_STATS["gp"]: self.gp_s,
                "compute_time_s": self.mma_s + self.gp_s,
            },
        )
        return BoundAnalyzer.dominant(result)

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
    name: str
    latency_s: float
    calls: int
    share: BoundShare

    @property
    def bound(self) -> str:
        return self.share.dominant()


@dataclass
class SweepPoint:
    batch_size: int
    latency_s: float
    classified: BoundShare
    components: BoundShare
    operators: tuple[OperatorSample, ...] = ()

    @property
    def bound(self) -> str:
        classified = self.classified.dominant()
        if classified != UNKNOWN_BOUND:
            return classified
        return self.components.dominant()


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

    def network_transitions(self) -> list[BoundTransition]:
        return bound_transitions(self.points, lambda point: point.bound)

    def top_operators(self) -> list[str]:
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
    """Rank operators by summed analytic latency across the sweep."""
    totals: dict[str, float] = {}
    for point in points:
        for operator in point.operators:
            totals[operator.name] = totals.get(operator.name, 0.0) + operator.latency_s
    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return [name for name, _total in ranked[: max(top_k, 0)]]


def operator_label(func: Any) -> str:
    """Short, stable operator name for charts."""
    packet = getattr(func, "overloadpacket", None)
    qualified = getattr(packet, "_qualified_op_name", None) if packet is not None else None
    if qualified:
        overload = getattr(func, "_overloadname", "") or ""
        if overload and overload != "default":
            return f"{qualified}.{overload}"
        return str(qualified)
    text = str(func)
    for prefix in ("torch.ops.", "aten::"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text


def aggregate_operators(events: Iterable[Any], model_name: str) -> tuple[OperatorSample, ...]:
    """Sum latency and bound components for each operator name."""
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        if getattr(event, "is_spill", False):
            continue
        result = event.perf_results.get(model_name)
        if result is None:
            continue
        name = operator_label(event.op_invoke_info.func)
        slot = grouped.setdefault(name, {"latency_s": 0.0, "calls": 0, "share": BoundShare()})
        slot["latency_s"] += float(result.execution_time_s)
        slot["calls"] += 1
        slot["share"] = slot["share"].add(BoundShare.from_result(result))
    samples = [
        OperatorSample(name=name, latency_s=slot["latency_s"], calls=slot["calls"], share=slot["share"])
        for name, slot in grouped.items()
    ]
    samples.sort(key=lambda sample: (-sample.latency_s, sample.name))
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
                    "network_share_percent": _network_share(sample.latency_s, point.latency_s),
                }
            )
        points.append(
            {
                "batch_size": point.batch_size,
                "latency_ms": point.latency_s * 1e3,
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
        "points": points,
    }


def render_html(report: LatencyReport) -> str:
    top = report.top_operators()
    network_summary = describe_bounds(report.points, lambda point: point.bound, "整网")
    batches = [point.batch_size for point in report.points]
    network_ms = [point.latency_s * 1e3 for point in report.points]
    network_bounds = [point.bound for point in report.points]
    body = [
        "<h1>Batch size 时延与 bound 趋势</h1>",
        "<p class='meta'>" + html.escape(_meta_line(report)) + "</p>",
        "<p class='summary'>" + html.escape(network_summary) + "</p>",
        "<p id='point-readout' class='readout'>点曲线上的圆点，这里显示该 batch 的时延、整网占比和 bound。</p>",
        _legend(),
        "<section><h2>整网时延</h2>",
        "<p class='note'>纵轴是解析模型调度后的设备执行时间。算子曲线是同名调用的 analytic 时间之和，多流重叠时可以大于整网时间。</p>",
        _latency_chart("整网", batches, network_ms, network_bounds, width=960, height=360),
        "<h3>Bound 构成</h3>",
        "<p class='note'>按算子主导资源归类后的时间占比，口径与 analytic_OpBound 一致。</p>",
        _mix_chart(batches, [point.classified for point in report.points]),
        _network_table(report),
        "</section>",
        "<section><h2>Top " + str(len(top)) + " 算子时延趋势</h2>",
        "<p class='note'>点一个算子标签只看它自己的曲线。圆点可读出时延和占整网调度时间的比例，"
        "底色是主导 bound。多流重叠时这个比例可以超过 100%。</p>",
        _operator_tags(top),
        _top_trend_chart(report),
        "<h3>单个算子</h3>",
        "<div class='grid' id='op-cards'>",
    ]
    for name in top:
        series = _operator_series(report.points, name)
        caption = describe_bounds(series, lambda sample: sample.bound, name) + _share_sentence(series)
        body.append(f"<article class='card op-card' data-op='{html.escape(name, quote=True)}'>")
        body.append("<h3>" + html.escape(name) + "</h3>")
        body.append("<p class='note'>" + html.escape(caption) + "</p>")
        body.append(
            _latency_chart(
                name,
                [sample.batch_size for sample in series],
                [sample.latency_s * 1e3 for sample in series],
                [sample.bound for sample in series],
                width=460,
                height=240,
                shares=[sample.network_share for sample in series],
            )
        )
        body.append("</article>")
    body.append("</div>")
    body.append(_top_latency_table(report))
    body.append("</section>")
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


def _network_share(operator_s: float, network_s: float) -> Optional[float]:
    """Operator analytic time as a percent of the scheduled network latency."""
    if network_s <= 0:
        return None
    return operator_s / network_s * 100.0


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
                _network_share(match.latency_s, point.latency_s),
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


def _meta_line(report: LatencyReport) -> str:
    return (
        f"{report.model_id} · {report.device} · {report.phase} · "
        f"query={report.query_length} context={report.context_length} · "
        f"TP={report.tp_size} · quant={report.quantize_linear} · {report.performance_model}"
    )


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
        latency_s = float(raw_point["latency_ms"]) / 1e3
        points.append(
            SweepPoint(
                batch_size=int(raw_point["batch_size"]),
                latency_s=latency_s,
                classified=_share_from_percent(raw_point.get("classified_percent", {}), latency_s),
                components=_share_from_percent(raw_point.get("components_percent", {}), latency_s),
                operators=tuple(operators),
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
    )


def _operator_tags(names: Sequence[str]) -> str:
    buttons = ["<button type='button' class='tag on' data-op='*'>全部</button>"]
    for index, name in enumerate(names):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        buttons.append(
            f"<button type='button' class='tag' data-op='{html.escape(name, quote=True)}'>"
            f"<i style='background:{color}'></i>{html.escape(name)}</button>"
        )
    return "<div class='tags' id='op-tags'>" + "".join(buttons) + "</div>"


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
    values_ms: Sequence[float],
    bounds: Sequence[str],
    *,
    width: int,
    height: int,
    shares: Optional[Sequence[Optional[float]]] = None,
) -> str:
    if not batches:
        return "<p class='note'>没有数据。</p>"
    pad_l, pad_r, pad_t, pad_b = 58, 16, 16, 36
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
    coords = []
    dots = []
    for index, (batch, value, bound) in enumerate(zip(batches, values_ms, bounds)):
        if value is None:
            continue
        x = x_at(index)
        y = y_at(value)
        coords.append(f"{x:.1f},{y:.1f}")
        share = None if shares is None or index >= len(shares) else shares[index]
        share_text = "" if share is None else f" · 占整网 {share:.1f}%"
        label = (
            f"{title} · batch {batch} · {value:.3f} ms · "
            f"{BOUND_LABELS.get(bound, bound)}{share_text}"
        )
        dots.append(
            _point_mark(
                x,
                y,
                4.5,
                BOUND_COLORS.get(bound, BOUND_COLORS[UNKNOWN_BOUND]),
                label,
            )
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
            shares.append(_network_share(match.latency_s, point.latency_s))
    return values, bounds, shares


def _top_trend_chart(report: LatencyReport) -> str:
    """One shared-axis chart of the top operators' latency versus batch size."""
    names = report.top_operators()
    batches = [point.batch_size for point in report.points]
    if not names or not batches:
        return "<p class='note'>没有算子数据。</p>"
    width, height = 960, 420
    pad_l, pad_r, pad_t, pad_b = 58, 16, 18, 36
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    prepared = [(name, *_series_values(report, name)) for name in names]
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
            f"<text x='{pad_l - 8}' y='{y + 4:.1f}' text-anchor='end' class='tick'>{value:.3g}</text>"
        )
    for index, batch in enumerate(batches):
        parts.append(
            f"<text x='{x_at(index):.1f}' y='{height - 12}' text-anchor='middle' class='tick'>{batch}</text>"
        )
    for series_index, (name, values, bounds, shares) in enumerate(prepared):
        color = SERIES_COLORS[series_index % len(SERIES_COLORS)]
        segment: list[str] = []
        dots: list[str] = []
        previous = None
        for index, value in enumerate(values):
            if value is None:
                if len(segment) > 1:
                    dots.append(
                        f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(segment)}'/>"
                    )
                segment = []
                previous = None
                continue
            x = x_at(index)
            y = y_at(value)
            segment.append(f"{x:.1f},{y:.1f}")
            bound = bounds[index]
            change = ""
            if previous is not None and previous != bound:
                change = f" · {BOUND_LABELS.get(previous, previous)} → {BOUND_LABELS.get(bound, bound)}"
            previous = bound
            share = shares[index]
            share_text = "" if share is None else f" · 占整网 {share:.1f}%"
            label = (
                f"{name} · batch {batches[index]} · {value:.3f} ms · "
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
        if len(segment) > 1:
            dots.insert(0, f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(segment)}'/>")
        parts.append(
            f"<g class='op-series' data-op='{html.escape(name, quote=True)}'>" + "".join(dots) + "</g>"
        )
    parts.append(f"<text x='{pad_l}' y='12' class='axis'>ms</text>")
    return (
        "<div id='op-compare'><svg viewBox='0 0 "
        + f"{width} {height}' role='img' aria-label='top operator latency trends'>"
        + "".join(parts)
        + "</svg></div>"
    )


def _top_latency_table(report: LatencyReport) -> str:
    names = report.top_operators()
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
            + html.escape(name)
            + "</td>"
            + "".join(cells)
            + "<td>"
            + html.escape(change_text)
            + "</td></tr>"
        )
    return (
        "<table class='ops'><caption>上面是毫秒，下面是占整网调度时间的比例。</caption><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _mix_chart(batches: Sequence[int], shares: Sequence[BoundShare]) -> str:
    if not batches:
        return ""
    width, height = 960, 180
    pad_l, pad_r, pad_t, pad_b = 58, 16, 10, 32
    slot = (width - pad_l - pad_r) / len(batches)
    bar_w = min(36, slot * 0.62)
    parts = []
    for index, (batch, share) in enumerate(zip(batches, shares)):
        percents = share.percent()
        x = pad_l + slot * index + (slot - bar_w) / 2
        y = pad_t
        plot_h = height - pad_t - pad_b
        for key in (COMPUTE_BOUND_GP, COMPUTE_BOUND_MMA, COMMUNICATION_BOUND, MEMORY_BOUND):
            fraction = percents.get(key, 0.0) / 100.0
            bar_h = plot_h * fraction
            y += bar_h
            if bar_h <= 0:
                continue
            parts.append(
                f"<rect x='{x:.1f}' y='{y - bar_h:.1f}' width='{bar_w:.1f}' height='{bar_h:.1f}' "
                f"fill='{BOUND_COLORS[key]}'>"
                f"<title>{html.escape(BOUND_LABELS[key])} {percents[key]:.1f}%</title></rect>"
            )
        parts.append(
            f"<text x='{x + bar_w / 2:.1f}' y='{height - 12}' text-anchor='middle' class='tick'>{batch}</text>"
        )
    return f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='bound mix'>" + "".join(parts) + "</svg>"


def _network_table(report: LatencyReport) -> str:
    rows = []
    previous = None
    for point in report.points:
        percents = point.classified.percent()
        changed = "" if previous is None or previous == point.bound else "切换"
        previous = point.bound
        rows.append(
            "<tr>"
            f"<td>{point.batch_size}</td>"
            f"<td>{point.latency_s * 1e3:.4f}</td>"
            f"<td>{html.escape(BOUND_LABELS.get(point.bound, point.bound))}</td>"
            f"<td>{percents[MEMORY_BOUND]:.1f}</td>"
            f"<td>{percents[COMMUNICATION_BOUND]:.1f}</td>"
            f"<td>{percents[COMPUTE_BOUND_MMA]:.1f}</td>"
            f"<td>{percents[COMPUTE_BOUND_GP]:.1f}</td>"
            f"<td>{changed}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>batch</th><th>整网 ms</th><th>主导 bound</th>"
        "<th>memory %</th><th>comm %</th><th>mma %</th><th>gp %</th><th></th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


_STYLE = """
body{margin:0;background:#f4f1ea;color:#1c1915;
font:16px/1.55 "Iowan Old Style","Palatino Linotype",Palatino,"Songti SC",serif}
body>*{max-width:1000px}
h1,h2,h3,p,section,div.legend{margin-left:auto;margin-right:auto}
h1{max-width:1000px;padding:1.6rem 1.2rem 0}
section{max-width:1000px;margin:0 auto 1.5rem;padding:0 1.2rem}
h1{font-size:1.8rem;font-weight:600;margin-bottom:.2rem}
h2{font-size:1.25rem;margin:1.4rem 0 .4rem}
h3{font-size:1rem;margin:.8rem 0 .2rem}
.meta,.note{color:#5e584f}
.summary{background:#fffcf7;border:1px solid #e3ddd2;border-radius:10px;padding:.8rem 1rem}
.readout{position:sticky;top:0;z-index:2;background:#1c1915;color:#fffcf7;border-radius:10px;padding:.55rem .8rem}
.point{cursor:pointer}
.point .hit{fill:transparent}
.point.on .dot{stroke:#1c1915;stroke-width:2.5}
.share{color:#5e584f;font:11px/1.3 ui-sans-serif,system-ui,sans-serif}
caption{caption-side:bottom;text-align:left;color:#5e584f;
font:12px/1.4 ui-sans-serif,system-ui,sans-serif;padding-top:.3rem}
.legend{display:flex;flex-wrap:wrap;gap:.8rem 1.2rem;padding:.4rem 1.2rem 1rem}
.tags{display:flex;flex-wrap:wrap;gap:.45rem;margin:.2rem 0 .8rem}
.tag{display:inline-flex;align-items:center;gap:.35rem;border:1px solid #e3ddd2;background:#fffcf7;
color:#1c1915;border-radius:999px;padding:.28rem .7rem;font:13px/1.3 ui-sans-serif,system-ui,sans-serif;cursor:pointer}
.tag i{width:.7rem;height:.7rem;border-radius:2px;display:inline-block}
.tag.on{background:#1c1915;color:#fffcf7;border-color:#1c1915}
.grid.solo{grid-template-columns:1fr}
[hidden]{display:none !important}
table.ops td.op,table.ops th:first-child{text-align:left}
.legend span{display:inline-flex;align-items:center;gap:.4rem}
.legend i{width:.85rem;height:.85rem;border-radius:2px;display:inline-block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:.8rem}
.card{background:#fffcf7;border:1px solid #e3ddd2;border-radius:10px;padding:.6rem .7rem .2rem}
svg{width:100%;height:auto;display:block}
.tick{font:11px/1 ui-sans-serif,system-ui,sans-serif;fill:#5e584f}
.marker{font:11px/1 ui-sans-serif,system-ui,sans-serif;fill:#1c1915}
.axis{font:11px/1 ui-sans-serif,system-ui,sans-serif;fill:#5e584f}
table{border-collapse:collapse;width:100%;background:#fffcf7;margin:.6rem 0 1rem}
th,td{text-align:right;padding:.35rem .5rem;border-bottom:1px solid #e3ddd2;font-variant-numeric:tabular-nums}
th:nth-child(3),td:nth-child(3),th:last-child,td:last-child{text-align:left}
th{font:12px/1.4 ui-sans-serif,system-ui,sans-serif;color:#5e584f;font-weight:600}
"""

_FILTER_SCRIPT = """
const tags=document.querySelectorAll('#op-tags .tag');
function selectOp(name){
  const all=name==='*';
  document.querySelectorAll('.op-series,.op-card,.op-row').forEach(el=>{
    el.hidden=!all && el.dataset.op!==name;
  });
  const compare=document.getElementById('op-compare');
  if(compare) compare.hidden=!all;
  const cards=document.getElementById('op-cards');
  if(cards) cards.classList.toggle('solo', !all);
  tags.forEach(btn=>btn.classList.toggle('on', btn.dataset.op===name));
}
tags.forEach(btn=>btn.addEventListener('click',()=>selectOp(btn.dataset.op)));
const readout=document.getElementById('point-readout');
function showPoint(el){
  document.querySelectorAll('.point.on').forEach(node=>node.classList.remove('on'));
  el.classList.add('on');
  if(readout) readout.textContent=el.dataset.tip || '';
}
document.querySelectorAll('.point').forEach(el=>{
  el.addEventListener('click',event=>{event.stopPropagation(); showPoint(el);});
  el.addEventListener('keydown',event=>{
    if(event.key==='Enter' || event.key===' '){event.preventDefault(); showPoint(el);}
  });
});
"""

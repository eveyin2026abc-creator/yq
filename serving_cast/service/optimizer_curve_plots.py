# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Optimizer curve plots: terminal ASCII throughput/QPS curves (plotext).

The terminal path relies on the optional ``plotext`` package, which exposes plotting
through module-level functions backed by shared canvas state. See
``_emit_terminal_optimizer_curve_ascii`` for concurrency/thread-safety notes.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from serving_cast.service.optimizer_summary import (
    render_cross_device_comparison,
    render_cross_hardware_disagg_decode,
    render_cross_hardware_disagg_prefill,
    render_cross_hardware_pd_ratio,
    render_hardware_profile_comparison,
)

logger = logging.getLogger(__name__)

_PALETTE = [
    (144, 238, 144),
    (200, 200, 200),
    (135, 206, 235),
    (255, 182, 193),
    (255, 160, 122),
    (221, 160, 221),
]

# Terminal canvas (plotext): shared internal canvas; not safe across overlapping calls /
# threads unless serialized externally (see _emit_terminal_optimizer_curve_ascii).
_TERMINAL_PLOT_COLS = 128
_TERMINAL_PLOT_ROWS = 38
_TERMINAL_MARKER = "dot"
_AXIS_PADDING_RATIO = 0.08
_DEVICE_MARKERS = ("+", "*", "x")
_LABEL_DIRECTIONS = (
    (0.0, 1.3),
    (1.1, 1.0),
    (-1.1, 1.0),
    (1.4, 0.0),
    (-1.4, 0.0),
    (0.0, -1.3),
    (1.1, -1.0),
    (-1.1, -1.0),
)
_BASE_CURVE_COLUMNS = ("concurrency", "token/s")
_PD_TPS_RENAME = {
    "parallel_d": "parallel",
    "concurrency_d": "concurrency",
    "tpot_d": "tpot",
}
_PREFILL_EMIT_KWARGS = {"chart2_x_col": "ttft", "chart2_x_label": "TTFT (ms)"}
_DECODE_EMIT_KWARGS = {"chart2_x_col": "tpot", "chart2_x_label": "TPOT (ms)"}


def _axis_metric_name(axis_label: str) -> str:
    return axis_label.split(" (", 1)[0].strip() or axis_label


def _parallel_label(parallel: str) -> str:
    s = str(parallel)
    return s if len(s) < 48 else s[:45] + "..."


def _short_device_label(device: str) -> str:
    name = str(device)
    if name.startswith("ATLAS_800_"):
        name = name.removeprefix("ATLAS_800_")
    return name if len(name) <= 28 else name[:25] + "..."


def _text_offsets_for_points(
    xs: list[float], ys: list[float]
) -> list[tuple[float, float, float, float]]:
    """Return scatter jitter and label offsets for each point."""
    group_counts: dict[tuple[float, float], int] = {}
    group_sizes: dict[tuple[float, float], int] = {}
    for x, y in zip(xs, ys):
        key = (round(float(x), 6), round(float(y), 6))
        group_sizes[key] = group_sizes.get(key, 0) + 1

    xspan = max(xs) - min(xs) if xs else 0.0
    yspan = max(ys) - min(ys) if ys else 0.0
    pad_x = max(xspan * 0.06, 1.0)
    pad_y = max(yspan * 0.08, max(ys) * 0.012 if ys else 1.0)

    offsets: list[tuple[float, float, float, float]] = []
    for x, y in zip(xs, ys):
        key = (round(float(x), 6), round(float(y), 6))
        idx = group_counts.get(key, 0)
        group_counts[key] = idx + 1
        group_n = group_sizes[key]
        dir_x, dir_y = _LABEL_DIRECTIONS[idx % len(_LABEL_DIRECTIONS)]
        label_dx = pad_x * dir_x
        label_dy = pad_y * dir_y
        scatter_dx = pad_x * 0.12 * (idx - (group_n - 1) / 2.0) if group_n > 1 else 0.0
        scatter_dy = pad_y * 0.10 * (idx - (group_n - 1) / 2.0) if group_n > 1 else 0.0
        offsets.append((scatter_dx, scatter_dy, label_dx, label_dy))
    return offsets


def _padded_axis_limits(values: list[float]) -> tuple[float, float] | None:
    nums = [float(v) for v in values if math.isfinite(float(v))]
    if not nums:
        return None

    lower = min(nums)
    upper = max(nums)
    span = upper - lower
    padding = span * _AXIS_PADDING_RATIO if span else max(abs(lower) * 0.1, 1.0)
    padded_lower = lower - padding
    return (max(0.0, padded_lower) if lower >= 0 else padded_lower, upper + padding)


def _sorted_sub_concurrency(
    curve_df: pd.DataFrame, parallel: str, tie_latency_col: str
) -> pd.DataFrame:
    sub = curve_df.loc[curve_df["parallel"].astype(str) == parallel]
    sort_cols = ["concurrency"]
    if "batch_size" in sub.columns:
        sub = sub.assign(_batch_sort=pd.to_numeric(sub["batch_size"], errors="coerce"))
        sort_cols.append("_batch_sort")
    sort_cols.append(tie_latency_col)
    sub = sub.sort_values(sort_cols)
    return sub.drop(columns=["_batch_sort"], errors="ignore")


def _sorted_sub_latency_axis(
    curve_df: pd.DataFrame, parallel: str, x_col: str
) -> pd.DataFrame:
    sub = curve_df.loc[curve_df["parallel"].astype(str) == parallel]
    sort_cols = [x_col, "concurrency"]
    if "batch_size" in sub.columns:
        sub = sub.assign(_batch_sort=pd.to_numeric(sub["batch_size"], errors="coerce"))
        sort_cols.insert(1, "_batch_sort")
    sub = sub.sort_values(sort_cols)
    return sub.drop(columns=["_batch_sort"], errors="ignore")


def _emit_terminal_optimizer_curve_ascii(
    curve_df: pd.DataFrame,
    title_prefix: str,
    *,
    chart2_x_col: str = "tpot",
    chart2_x_label: str = "TPOT (ms)",
    y_axis_label: str = "Throughput (token/s)",
) -> None:
    """Print throughput (or QPS) curves as terminal ASCII using plotext."""
    try:
        import plotext as plx
    except ImportError:
        logger.warning("plotext is not installed; skipping terminal curve plots.")
        return

    parallels = sorted(curve_df["parallel"].astype(str).unique())
    if not parallels:
        return
    y_metric = _axis_metric_name(y_axis_label)

    def _draw_chart(
        title: str,
        x_col: str,
        x_label: str,
        sort_fn: Callable[[pd.DataFrame, str], pd.DataFrame],
    ) -> None:
        plx.plot_size(_TERMINAL_PLOT_COLS, _TERMINAL_PLOT_ROWS)
        plx.theme("clear")
        x_all: list[float] = []
        y_all: list[float] = []
        for idx, parallel in enumerate(parallels):
            sub = sort_fn(curve_df, parallel)
            if sub.empty:
                continue
            points = pd.DataFrame(
                {
                    "x": pd.to_numeric(sub[x_col], errors="coerce"),
                    "y": pd.to_numeric(sub["token/s"], errors="coerce"),
                }
            ).dropna()
            if points.empty:
                continue
            xv = points["x"].tolist()
            yv = points["y"].tolist()
            x_all.extend(xv)
            y_all.extend(yv)
            plx.plot(
                xv,
                yv,
                label=_parallel_label(parallel),
                color=_PALETTE[idx % len(_PALETTE)],
                marker=_TERMINAL_MARKER,
            )
        xlim = _padded_axis_limits(x_all)
        ylim = _padded_axis_limits(y_all)
        if xlim is not None:
            plx.xlim(*xlim)
        if ylim is not None:
            plx.ylim(*ylim)
        plx.title(f"{title_prefix}: {title}")
        plx.xlabel(x_label)
        plx.ylabel(y_axis_label)
        plx.grid(False)
        try:
            buf = plx.build()
        except Exception:
            logger.exception("plotext failed to build chart: %s", title)
            buf = ""
        finally:
            plx.clear_data()
        if buf:
            print("\n" + buf + "\n")

    try:
        chart_specs = (
            (
                f"{y_metric} vs concurrency",
                "concurrency",
                "Concurrency",
                lambda df, p: _sorted_sub_concurrency(df, p, chart2_x_col),
            ),
            (
                f"{y_metric} vs {chart2_x_label.split()[0]}",
                chart2_x_col,
                chart2_x_label,
                lambda df, p: _sorted_sub_latency_axis(df, p, chart2_x_col),
            ),
        )
        for title, x_col, x_label, sort_fn in chart_specs:
            _draw_chart(title, x_col, x_label, sort_fn)
    except Exception:
        logger.exception("Terminal ASCII optimizer curves failed.")


def _emit_multi_device_scatter_ascii(
    curve_df: pd.DataFrame,
    title_prefix: str,
    *,
    chart2_x_col: str = "tpot",
    chart2_x_label: str = "TPOT (ms)",
    y_axis_label: str = "Throughput (token/s)",
) -> None:
    """Print one scatter point per device with inline device labels."""
    try:
        import plotext as plx
    except ImportError:
        logger.warning("plotext is not installed; skipping terminal curve plots.")
        return

    if curve_df.empty:
        return

    y_metric = _axis_metric_name(y_axis_label)
    chart_specs = (
        (f"{y_metric} vs concurrency", "concurrency", "Concurrency"),
        (f"{y_metric} vs {chart2_x_label.split()[0]}", chart2_x_col, chart2_x_label),
    )

    def _draw_chart(title: str, x_col: str, x_label: str) -> None:
        points = curve_df[["parallel", x_col, "token/s"]].copy()
        points[x_col] = pd.to_numeric(points[x_col], errors="coerce")
        points["token/s"] = pd.to_numeric(points["token/s"], errors="coerce")
        points = points.dropna(subset=["parallel", x_col, "token/s"])
        if points.empty:
            return

        xs = points[x_col].tolist()
        ys = points["token/s"].tolist()
        labels = [_short_device_label(name) for name in points["parallel"].astype(str)]
        point_offsets = _text_offsets_for_points(xs, ys)

        scatter_xs: list[float] = []
        scatter_ys: list[float] = []
        label_xs: list[float] = []
        label_ys: list[float] = []

        plx.plot_size(_TERMINAL_PLOT_COLS, _TERMINAL_PLOT_ROWS)
        plx.theme("clear")
        for idx, (x, y, label) in enumerate(zip(xs, ys, labels)):
            color = _PALETTE[idx % len(_PALETTE)]
            marker = _DEVICE_MARKERS[idx % len(_DEVICE_MARKERS)]
            scatter_dx, scatter_dy, label_dx, label_dy = point_offsets[idx]
            sx = x + scatter_dx
            sy = y + scatter_dy
            lx = sx + label_dx
            ly = sy + label_dy
            scatter_xs.extend([sx, x])
            scatter_ys.extend([sy, y])
            label_xs.append(lx)
            label_ys.append(ly)
            plx.scatter(
                [sx],
                [sy],
                label=f"[{marker}] {label}",
                color=color,
                marker=marker,
            )
            plx.text(f"[{marker}] {label}", lx, ly, color=color)

        xlim = _padded_axis_limits(xs + scatter_xs + label_xs)
        ylim = _padded_axis_limits(ys + scatter_ys + label_ys)
        if xlim is not None:
            plx.xlim(*xlim)
        if ylim is not None:
            plx.ylim(*ylim)
        plx.title(f"{title_prefix}: {title}")
        plx.xlabel(x_label)
        plx.ylabel(y_axis_label)
        plx.grid(False)
        try:
            buf = plx.build()
        except Exception:
            logger.exception("plotext failed to build multi-device scatter: %s", title)
            buf = ""
        finally:
            plx.clear_data()
        if buf:
            print("\n" + buf + "\n")

    try:
        for title, x_col, x_label in chart_specs:
            _draw_chart(title, x_col, x_label)
    except Exception:
        logger.exception("Terminal ASCII multi-device scatter plots failed.")


def _memory_filter(work: pd.DataFrame) -> pd.DataFrame:
    for mem_col in ("memory_left_gb", "device_memory_available_gb"):
        if mem_col in work.columns:
            mem = pd.to_numeric(work[mem_col], errors="coerce")
            work = work.loc[mem.isna() | (mem > 0)]
            break
    return work


def _require_columns(df: pd.DataFrame, required: set[str], message: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{message}: {sorted(missing)}")


def _sort_curve_df(work: pd.DataFrame) -> pd.DataFrame:
    if work.empty:
        return work

    sort_keys = ["parallel", "concurrency"]
    if "batch_size" in work.columns:
        work = work.assign(
            _batch_sort=pd.to_numeric(work["batch_size"], errors="coerce")
        )
        sort_keys.append("_batch_sort")
    sort_keys.append("token/s")
    return (
        work.sort_values(sort_keys)
        .reset_index(drop=True)
        .drop(columns=["_batch_sort"], errors="ignore")
    )


def _prepare_base_curve_df(
    df: pd.DataFrame,
    *,
    latency_col: str,
    missing_message: str,
) -> pd.DataFrame:
    required = {"parallel", latency_col, *_BASE_CURVE_COLUMNS}
    _require_columns(df, required, missing_message)

    work = df.copy()
    for col in (*_BASE_CURVE_COLUMNS, latency_col):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work.dropna(subset=["parallel", *_BASE_CURVE_COLUMNS, latency_col])


def _prepare_curve_df(
    df: pd.DataFrame,
    _ttft_limit: float | None,
    _tpot_limit: float | None,
) -> pd.DataFrame:
    """Aggregation rows: token/s vs concurrency / TPOT (latency column ``tpot``)."""
    return _prepare_latency_curve_df(
        df,
        latency_col="tpot",
        missing_message="DataFrame missing columns for curve plots",
    )


def _prepare_disagg_prefill_curve_df(
    df: pd.DataFrame,
    _ttft_limit: float | None,
) -> pd.DataFrame:
    """Disagg Prefill sweep: token/s vs concurrency / TTFT."""
    return _prepare_latency_curve_df(
        df,
        latency_col="ttft",
        missing_message="Prefill curve plot missing columns",
    )


def _prepare_latency_curve_df(
    df: pd.DataFrame,
    *,
    latency_col: str,
    missing_message: str,
) -> pd.DataFrame:
    work = _prepare_base_curve_df(
        df,
        latency_col=latency_col,
        missing_message=missing_message,
    )
    work = _memory_filter(work)
    return _sort_curve_df(work)


def plot_concurrency_curves_from_optimizer_summaries(
    results: list,
    *,
    basename_prefix: str,
    ttft_limit: float | None = None,
    tpot_limit: float | None = None,
) -> bool:
    """Merge aggregation summary frames and print terminal curves."""
    dfs = [
        df
        for r in results
        if (df := r.get_summary_df()) is not None and not df.empty
    ]
    if not dfs:
        return False
    merged = pd.concat(dfs, ignore_index=True)
    return plot_concurrency_optimizer_curves(
        merged,
        basename_prefix=basename_prefix,
        ttft_limit=ttft_limit,
        tpot_limit=tpot_limit,
    )


def plot_concurrency_optimizer_curves(
    df: pd.DataFrame,
    *,
    basename_prefix: str,
    ttft_limit: float | None = None,
    tpot_limit: float | None = None,
) -> bool:
    """Aggregation mode terminal curves (token/s vs concurrency / TPOT)."""
    try:
        curve_df = _prepare_curve_df(df, ttft_limit, tpot_limit)
    except ValueError as exc:
        logger.warning("Skipping concurrency curve plots: %s", exc)
        return False

    return _emit_curve_df(
        curve_df,
        title_prefix=str(basename_prefix).strip()[:160] or "optimizer",
        skip_label="concurrency curve plots",
    )


def _emit_curve_df(
    curve_df: pd.DataFrame,
    *,
    title_prefix: str,
    skip_label: str,
    emit_kwargs: dict[str, str] | None = None,
) -> bool:
    if curve_df.empty:
        logger.warning("Skipping %s: no rows after filtering.", skip_label)
        return False

    _emit_terminal_optimizer_curve_ascii(
        curve_df,
        title_prefix=title_prefix,
        **(emit_kwargs or {}),
    )
    return True


def _emit_prepared_curve(
    prepare_curve: Callable[[], pd.DataFrame],
    *,
    title_prefix: str,
    skip_label: str,
    emit_kwargs: dict[str, str],
) -> bool:
    try:
        curve_df = prepare_curve()
    except ValueError as exc:
        logger.warning("Skipping %s: %s", skip_label, exc)
        return False
    return _emit_curve_df(
        curve_df,
        title_prefix=title_prefix,
        skip_label=skip_label,
        emit_kwargs=emit_kwargs,
    )


def plot_disagg_terminal_curves(
    results: list,
    *,
    basename_prefix: str,
    ttft_limit: float | None,
    tpot_limit: float | None,
) -> bool:
    """Terminal curves for disaggregation Prefill (TTFT x-axis) and/or Decode (TPOT)."""
    any_ok = False
    base = str(basename_prefix).strip()[:140] or "optimizer"

    for idx, res in enumerate(results):
        df = res.get_summary_df()
        if df is None or df.empty:
            continue
        dc = getattr(res, "data_config", None)
        if dc is None:
            continue

        prefill = dc.ttft_limits is not None and dc.tpot_limits is None
        decode = dc.tpot_limits is not None and dc.ttft_limits is None

        if prefill:
            any_ok |= _emit_prepared_curve(
                lambda df=df: _prepare_disagg_prefill_curve_df(df, ttft_limit),
                title_prefix=f"{base}_disagg_prefill_{idx}",
                skip_label="prefill concurrency",
                emit_kwargs=_PREFILL_EMIT_KWARGS,
            )

        elif decode:
            any_ok |= _emit_prepared_curve(
                lambda df=df: _prepare_curve_df(df, ttft_limit, tpot_limit),
                title_prefix=f"{base}_disagg_decode_{idx}",
                skip_label="decode concurrency",
                emit_kwargs=_DECODE_EMIT_KWARGS,
            )

    return any_ok


def _pd_tps_curve_df(
    pd_df: pd.DataFrame,
) -> pd.DataFrame:
    source_cols = tuple(_PD_TPS_RENAME)
    _require_columns(pd_df, set(source_cols), "PD TPS curve plot missing columns")
    work = pd_df[list(source_cols)].drop_duplicates().rename(columns=_PD_TPS_RENAME)
    work["tpot"] = pd.to_numeric(work["tpot"], errors="coerce")
    work["concurrency"] = pd.to_numeric(work["concurrency"], errors="coerce")
    work = work.loc[work["tpot"] > 0]
    work["token/s"] = (
        pd.to_numeric(work["concurrency"], errors="coerce") / work["tpot"] * 1000
    )
    return work


def plot_pd_ratio_terminal_curves(
    pd_df: pd.DataFrame,
    *,
    basename_prefix: str,
    ttft_limit: float | None,
    tpot_limit: float | None,
) -> bool:
    """Terminal curves for PD-ratio grid: TPS vs concurrency and TPOT."""
    if pd_df.empty:
        return False

    return _emit_prepared_curve(
        lambda: _prepare_curve_df(_pd_tps_curve_df(pd_df), ttft_limit, tpot_limit),
        title_prefix=f"{str(basename_prefix).strip()[:120] or 'optimizer'}_pd_decode_tps",
        skip_label="PD TPS",
        emit_kwargs=_DECODE_EMIT_KWARGS,
    )


def _best_rows_curve_df(rows: list[dict], mode: str) -> pd.DataFrame:
    records = []
    for row in rows:
        if mode == "pd_ratio":
            record = {
                "parallel": row.get("device"),
                "concurrency": row.get("concurrency_d"),
                "token/s": row.get("decode_tps"),
                "tpot": row.get("tpot_d"),
            }
        else:
            record = {
                "parallel": row.get("device"),
                "concurrency": row.get("concurrency"),
                "token/s": row.get("throughput_tps"),
                "tpot": row.get("tpot_ms"),
            }
        records.append(record)

    if not records:
        return pd.DataFrame()
    return _prepare_curve_df(pd.DataFrame.from_records(records), None, None)


def plot_multi_device_best_terminal_curves(
    rows: list[dict],
    *,
    title_prefix: str,
    mode: str = "throughput",
) -> bool:
    """Terminal scatter plots for best case of each device in multi-device runs."""
    try:
        curve_df = _best_rows_curve_df(rows, mode)
    except ValueError as exc:
        logger.warning("Skipping multi-device best curve plots: %s", exc)
        return False

    if curve_df.empty:
        logger.warning("Skipping multi-device best curve plots: no rows after filtering.")
        return False

    _emit_multi_device_scatter_ascii(
        curve_df,
        title_prefix=title_prefix,
        **_DECODE_EMIT_KWARGS,
    )
    return True


@dataclass
class MultiDeviceComparisonRows:
    aggregation: list[dict] = field(default_factory=list)
    pd_ratio: list[dict] = field(default_factory=list)
    disagg_prefill: list[dict] = field(default_factory=list)
    disagg_decode: list[dict] = field(default_factory=list)


def _first_non_empty_summary_df(results: list):
    for res in results:
        summary_df = res.get_summary_df()
        if summary_df is not None and not summary_df.empty:
            return summary_df
    return None


def _plot_single_device_optimizer_curves(
    results: list,
    args,
    *,
    basename_prefix: str,
) -> None:
    """Dispatch terminal curve plotting for the active optimizer mode."""
    plot_kwargs = {
        "basename_prefix": basename_prefix,
        "ttft_limit": args.ttft_limits,
        "tpot_limit": args.tpot_limits,
    }

    if args.enable_optimize_prefill_decode_ratio:
        summary_df = _first_non_empty_summary_df(results)
        if summary_df is not None:
            plot_pd_ratio_terminal_curves(summary_df, **plot_kwargs)
        return

    if args.disagg:
        plot_disagg_terminal_curves(results, **plot_kwargs)
        return

    plot_concurrency_curves_from_optimizer_summaries(results, **plot_kwargs)


def _collect_cross_hardware_row(
    rows: MultiDeviceComparisonRows,
    res,
    profile_name: str,
    args,
) -> None:
    if args.disagg:
        collectors = (
            (res.collect_disagg_prefill_row, rows.disagg_prefill),
            (res.collect_disagg_decode_row, rows.disagg_decode),
        )
    elif args.enable_optimize_prefill_decode_ratio:
        collectors = ((res.collect_pd_ratio_comparison_row, rows.pd_ratio),)
    else:
        collectors = ((res.collect_comparison_row, rows.aggregation),)

    for collect, target in collectors:
        row = collect(profile_name)
        if row:
            target.append(row)


def run_multi_device_loop(
    args,
    device_targets: list[str],
    *,
    plot_curves_allowed: bool,
    logger: logging.Logger,
) -> MultiDeviceComparisonRows:
    """Run ParallelRunner per device and collect cross-hardware rows."""
    from serving_cast.parallel_runner import ParallelRunner

    rows = MultiDeviceComparisonRows()
    multi_hw = len(device_targets) > 1

    for profile_name in device_targets:
        args.device = profile_name
        logger.info("Hardware profile: %s", profile_name)
        tasks = ParallelRunner(args)

        results = (
            tasks.run_agg()
            if not args.enable_optimize_prefill_decode_ratio and not args.disagg
            else tasks.run_disagg()
        )

        for res in results:
            res.report_final_result(args, silent=False)
            if multi_hw:
                _collect_cross_hardware_row(rows, res, profile_name, args)

        if plot_curves_allowed:
            _plot_single_device_optimizer_curves(
                results,
                args,
                basename_prefix=f"{profile_name}_{args.model_id}",
            )

    return rows


def render_cross_hardware_summary(
    args,
    device_targets: list[str],
    rows: MultiDeviceComparisonRows,
    *,
    logger: logging.Logger,
) -> None:
    """Print cross-hardware comparison tables for multi-device runs."""
    if len(device_targets) <= 1:
        return

    hw_profile_txt = render_hardware_profile_comparison(device_targets)
    if hw_profile_txt:
        print(hw_profile_txt)

    if args.disagg:
        for rendered in (
            render_cross_hardware_disagg_prefill(rows.disagg_prefill),
            render_cross_hardware_disagg_decode(rows.disagg_decode),
        ):
            if rendered:
                print(rendered)
        plot_multi_device_best_terminal_curves(
            rows.disagg_decode,
            title_prefix="multi_device_disagg_decode_best",
        )
        if not rows.disagg_prefill and not rows.disagg_decode:
            logger.warning(
                "No rows available for cross-hardware disaggregation comparison "
                "(all runs empty or limits omitted)."
            )
        return

    render_fn, table_rows, warning = (
        (
            render_cross_hardware_pd_ratio,
            rows.pd_ratio,
            "No rows available for cross-hardware PD ratio comparison "
            "(all runs empty or filtered out).",
        )
        if args.enable_optimize_prefill_decode_ratio
        else (
            render_cross_device_comparison,
            rows.aggregation,
            "No rows available for cross-hardware comparison (all runs empty).",
        )
    )
    rendered = render_fn(table_rows)
    if rendered:
        print(rendered)
        plot_multi_device_best_terminal_curves(
            table_rows,
            title_prefix=(
                "multi_device_pd_ratio_best"
                if args.enable_optimize_prefill_decode_ratio
                else "multi_device_best"
            ),
            mode=(
                "pd_ratio"
                if args.enable_optimize_prefill_decode_ratio
                else "throughput"
            ),
        )
    elif not table_rows:
        logger.warning(warning)

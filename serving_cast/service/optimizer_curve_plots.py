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
from collections.abc import Callable

import pandas as pd

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
_BASE_CURVE_COLUMNS = ("concurrency", "token/s")
_PD_PREFILL_RENAME = {
    "parallel_p": "parallel",
    "concurrency_p": "concurrency",
    "p_qps": "token/s",
    "ttft_p": "ttft",
}
_PD_DECODE_RENAME = {
    "parallel_d": "parallel",
    "concurrency_d": "concurrency",
    "d_qps": "token/s",
    "tpot_d": "tpot",
}


def _axis_metric_name(axis_label: str) -> str:
    return axis_label.split(" (", 1)[0].strip() or axis_label


def _parallel_label(parallel: str) -> str:
    s = str(parallel)
    return s if len(s) < 48 else s[:45] + "..."


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
        for idx, parallel in enumerate(parallels):
            sub = sort_fn(curve_df, parallel)
            if sub.empty:
                continue
            xv = pd.to_numeric(sub[x_col], errors="coerce").tolist()
            yv = pd.to_numeric(sub["token/s"], errors="coerce").tolist()
            plx.plot(
                xv,
                yv,
                label=_parallel_label(parallel),
                color=_PALETTE[idx % len(_PALETTE)],
                marker=_TERMINAL_MARKER,
            )
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


def _filter_numeric_limit(
    work: pd.DataFrame,
    column: str,
    limit: float | None,
    *,
    skip_if_all_nan: bool = False,
) -> pd.DataFrame:
    if limit is None or column not in work.columns:
        return work
    values = pd.to_numeric(work[column], errors="coerce")
    if skip_if_all_nan and not values.notna().any():
        return work
    return work.loc[values.fillna(float("inf")) <= float(limit)]


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
    ttft_limit: float | None,
    tpot_limit: float | None,
) -> pd.DataFrame:
    """Aggregation rows: token/s vs concurrency / TPOT (latency column ``tpot``)."""
    work = _prepare_base_curve_df(
        df,
        latency_col="tpot",
        missing_message="DataFrame missing columns for curve plots",
    )
    work = _filter_numeric_limit(
        work, "ttft", ttft_limit, skip_if_all_nan=True
    )
    work = _filter_numeric_limit(work, "tpot", tpot_limit)
    work = _memory_filter(work)
    return _sort_curve_df(work)


def _prepare_disagg_prefill_curve_df(
    df: pd.DataFrame,
    ttft_limit: float | None,
) -> pd.DataFrame:
    """Disagg Prefill sweep: token/s vs concurrency / TTFT."""
    work = _prepare_base_curve_df(
        df,
        latency_col="ttft",
        missing_message="Prefill curve plot missing columns",
    )
    work = _filter_numeric_limit(work, "ttft", ttft_limit)
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
                emit_kwargs={"chart2_x_col": "ttft", "chart2_x_label": "TTFT (ms)"},
            )

        elif decode:
            any_ok |= _emit_prepared_curve(
                lambda df=df: _prepare_curve_df(df, ttft_limit, tpot_limit),
                title_prefix=f"{base}_disagg_decode_{idx}",
                skip_label="decode concurrency",
                emit_kwargs={"chart2_x_col": "tpot", "chart2_x_label": "TPOT (ms)"},
            )

    return any_ok


def _pd_side_curve_df(
    pd_df: pd.DataFrame,
    *,
    source_cols: tuple[str, ...],
    rename_cols: dict[str, str],
) -> pd.DataFrame | None:
    if not set(source_cols) <= set(pd_df.columns):
        return None
    return pd_df[list(source_cols)].drop_duplicates().rename(columns=rename_cols)


def _pd_curve_specs(
    pd_df: pd.DataFrame,
    ttft_limit: float | None,
    tpot_limit: float | None,
):
    p_norm = _pd_side_curve_df(
        pd_df,
        source_cols=tuple(_PD_PREFILL_RENAME),
        rename_cols=_PD_PREFILL_RENAME,
    )
    if p_norm is not None:
        yield (
            lambda df=p_norm: _prepare_disagg_prefill_curve_df(df, ttft_limit),
            "pd_prefill_qps",
            "PD Prefill-side",
            {
                "chart2_x_col": "ttft",
                "chart2_x_label": "TTFT (ms)",
                "y_axis_label": "P QPS (req/s)",
            },
        )

    d_norm = _pd_side_curve_df(
        pd_df,
        source_cols=tuple(_PD_DECODE_RENAME),
        rename_cols=_PD_DECODE_RENAME,
    )
    if d_norm is not None:
        yield (
            lambda df=d_norm: _prepare_curve_df(df, ttft_limit, tpot_limit),
            "pd_decode_qps",
            "PD Decode-side",
            {
                "chart2_x_col": "tpot",
                "chart2_x_label": "TPOT (ms)",
                "y_axis_label": "D QPS (req/s)",
            },
        )


def plot_pd_ratio_terminal_curves(
    pd_df: pd.DataFrame,
    *,
    basename_prefix: str,
    ttft_limit: float | None,
    tpot_limit: float | None,
) -> bool:
    """Terminal curves for PD-ratio grid: P/D QPS vs concurrency and latency."""
    if pd_df.empty:
        return False

    base = str(basename_prefix).strip()[:120] or "optimizer"
    any_ok = False

    for prepare_curve, title_suffix, skip_label, emit_kwargs in _pd_curve_specs(
        pd_df, ttft_limit, tpot_limit
    ):
        any_ok |= _emit_prepared_curve(
            prepare_curve,
            title_prefix=f"{base}_{title_suffix}",
            skip_label=skip_label,
            emit_kwargs=emit_kwargs,
        )

    return any_ok

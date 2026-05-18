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

    def _one_chart(
        title: str,
        x_col: str,
        x_label: str,
        y_label: str,
        sort_fn,
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
        plx.ylabel(y_label)
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
        _one_chart(
            f"{y_metric} vs concurrency",
            "concurrency",
            "Concurrency",
            y_axis_label,
            lambda df, p: _sorted_sub_concurrency(df, p, chart2_x_col),
        )
        _one_chart(
            f"{y_metric} vs {chart2_x_label.split()[0]}",
            chart2_x_col,
            chart2_x_label,
            y_axis_label,
            lambda df, p: _sorted_sub_latency_axis(df, p, chart2_x_col),
        )
    except Exception:
        logger.exception("Terminal ASCII optimizer curves failed.")


def _memory_filter(work: pd.DataFrame) -> pd.DataFrame:
    for mem_col in ("memory_left_gb", "device_memory_available_gb"):
        if mem_col in work.columns:
            mem = pd.to_numeric(work[mem_col], errors="coerce")
            work = work.loc[mem.isna() | (mem > 0)]
            break
    return work


def _prepare_curve_df(
    df: pd.DataFrame,
    ttft_limit: float | None,
    tpot_limit: float | None,
) -> pd.DataFrame:
    """Aggregation rows: token/s vs concurrency / TPOT (latency column ``tpot``)."""
    required = {"concurrency", "parallel", "token/s", "tpot"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns for curve plots: {sorted(missing)}")

    work = df.copy()
    work["concurrency"] = pd.to_numeric(work["concurrency"], errors="coerce")
    work["token/s"] = pd.to_numeric(work["token/s"], errors="coerce")
    work["tpot"] = pd.to_numeric(work["tpot"], errors="coerce")
    work = work.dropna(subset=["concurrency", "token/s", "tpot"])

    if ttft_limit is not None and "ttft" in work.columns:
        ttft_num = pd.to_numeric(work["ttft"], errors="coerce")
        if ttft_num.notna().any():
            work = work.loc[ttft_num.fillna(float("inf")) <= float(ttft_limit)]

    if tpot_limit is not None:
        work = work.loc[
            pd.to_numeric(work["tpot"], errors="coerce").fillna(float("inf"))
            <= float(tpot_limit)
        ]

    work = _memory_filter(work)

    if work.empty:
        return work

    sort_keys = ["parallel", "concurrency"]
    if "batch_size" in work.columns:
        work = work.assign(
            _batch_sort=pd.to_numeric(work["batch_size"], errors="coerce")
        )
        sort_keys.append("_batch_sort")
    sort_keys.append("token/s")
    work = work.sort_values(sort_keys).reset_index(drop=True)
    if "_batch_sort" in work.columns:
        work = work.drop(columns=["_batch_sort"])
    return work


def _prepare_disagg_prefill_curve_df(
    df: pd.DataFrame,
    ttft_limit: float | None,
) -> pd.DataFrame:
    """Disagg Prefill sweep: token/s vs concurrency / TTFT."""
    required = {"concurrency", "parallel", "token/s", "ttft"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prefill curve plot missing columns: {sorted(missing)}")

    work = df.copy()
    work["concurrency"] = pd.to_numeric(work["concurrency"], errors="coerce")
    work["token/s"] = pd.to_numeric(work["token/s"], errors="coerce")
    work["ttft"] = pd.to_numeric(work["ttft"], errors="coerce")
    work = work.dropna(subset=["concurrency", "token/s", "ttft"])

    if ttft_limit is not None:
        work = work.loc[
            pd.to_numeric(work["ttft"], errors="coerce").fillna(float("inf"))
            <= float(ttft_limit)
        ]

    work = _memory_filter(work)
    if work.empty:
        return work

    sort_keys = ["parallel", "concurrency"]
    if "batch_size" in work.columns:
        work = work.assign(
            _batch_sort=pd.to_numeric(work["batch_size"], errors="coerce")
        )
        sort_keys.append("_batch_sort")
    sort_keys.append("token/s")
    work = work.sort_values(sort_keys).reset_index(drop=True)
    if "_batch_sort" in work.columns:
        work = work.drop(columns=["_batch_sort"])
    return work


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

    if curve_df.empty:
        logger.warning(
            "Skipping concurrency curve plots: no rows after filtering "
            "(check TTFT/TPOT limits or optimizer output)."
        )
        return False

    parallels = sorted(curve_df["parallel"].astype(str).unique())
    if not parallels:
        return False

    title = str(basename_prefix).strip()[:160] or "optimizer"
    _emit_terminal_optimizer_curve_ascii(curve_df, title_prefix=title)
    return True


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
            try:
                curve_df = _prepare_disagg_prefill_curve_df(df, ttft_limit)
            except ValueError as exc:
                logger.warning("Skipping prefill curve plots: %s", exc)
                curve_df = pd.DataFrame()

            if curve_df.empty:
                logger.warning(
                    "Skipping prefill concurrency curves: no rows after filtering."
                )
            else:
                suffix = f"{base}_disagg_prefill_{idx}"
                _emit_terminal_optimizer_curve_ascii(
                    curve_df,
                    title_prefix=suffix,
                    chart2_x_col="ttft",
                    chart2_x_label="TTFT (ms)",
                )
                any_ok = True

        elif decode:
            try:
                curve_df = _prepare_curve_df(df, ttft_limit, tpot_limit)
            except ValueError as exc:
                logger.warning("Skipping decode curve plots: %s", exc)
                curve_df = pd.DataFrame()

            if curve_df.empty:
                logger.warning(
                    "Skipping decode concurrency curves: no rows after filtering."
                )
            else:
                suffix = f"{base}_disagg_decode_{idx}"
                _emit_terminal_optimizer_curve_ascii(
                    curve_df,
                    title_prefix=suffix,
                    chart2_x_col="tpot",
                    chart2_x_label="TPOT (ms)",
                )
                any_ok = True

    return any_ok


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

    prefill_cols = ["parallel_p", "concurrency_p", "p_qps", "ttft_p"]
    if set(prefill_cols) <= set(pd_df.columns):
        sub_p = pd_df[prefill_cols].drop_duplicates()
        p_norm = sub_p.rename(
            columns={
                "parallel_p": "parallel",
                "concurrency_p": "concurrency",
                "p_qps": "token/s",
                "ttft_p": "ttft",
            }
        )
        try:
            curve_p = _prepare_disagg_prefill_curve_df(p_norm, ttft_limit)
        except ValueError as exc:
            logger.warning("Skipping PD Prefill-side curves: %s", exc)
            curve_p = pd.DataFrame()

        if curve_p.empty:
            logger.warning(
                "Skipping PD Prefill-side curves: no rows after filtering."
            )
        else:
            _emit_terminal_optimizer_curve_ascii(
                curve_p,
                title_prefix=f"{base}_pd_prefill_qps",
                chart2_x_col="ttft",
                chart2_x_label="TTFT (ms)",
                y_axis_label="P QPS (req/s)",
            )
            any_ok = True

    decode_cols = ["parallel_d", "concurrency_d", "d_qps", "tpot_d"]
    if set(decode_cols) <= set(pd_df.columns):
        sub_d = pd_df[decode_cols].drop_duplicates()
        d_norm = sub_d.rename(
            columns={
                "parallel_d": "parallel",
                "concurrency_d": "concurrency",
                "d_qps": "token/s",
                "tpot_d": "tpot",
            }
        )
        try:
            curve_d = _prepare_curve_df(d_norm, ttft_limit, tpot_limit)
        except ValueError as exc:
            logger.warning("Skipping PD Decode-side curves: %s", exc)
            curve_d = pd.DataFrame()

        if curve_d.empty:
            logger.warning(
                "Skipping PD Decode-side curves: no rows after filtering."
            )
        else:
            _emit_terminal_optimizer_curve_ascii(
                curve_d,
                title_prefix=f"{base}_pd_decode_qps",
                chart2_x_col="tpot",
                chart2_x_label="TPOT (ms)",
                y_axis_label="D QPS (req/s)",
            )
            any_ok = True

    return any_ok

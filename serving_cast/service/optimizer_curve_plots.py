# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Optimizer curve plots: terminal ASCII throughput vs concurrency / TPOT (plotext).

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
# Single small marker keeps emphasis on line segments between sorted points.
_TERMINAL_MARKER = "dot"


def _parallel_label(parallel: str) -> str:
    s = str(parallel)
    return s if len(s) < 48 else s[:45] + "..."


def _sorted_sub_concurrency(curve_df: pd.DataFrame, parallel: str) -> pd.DataFrame:
    sub = curve_df.loc[curve_df["parallel"].astype(str) == parallel]
    sort_cols = ["concurrency"]
    if "batch_size" in sub.columns:
        sub = sub.assign(_batch_sort=pd.to_numeric(sub["batch_size"], errors="coerce"))
        sort_cols.append("_batch_sort")
    sort_cols.append("tpot")
    sub = sub.sort_values(sort_cols)
    return sub.drop(columns=["_batch_sort"], errors="ignore")


def _sorted_sub_tpot(curve_df: pd.DataFrame, parallel: str) -> pd.DataFrame:
    sub = curve_df.loc[curve_df["parallel"].astype(str) == parallel]
    sort_cols = ["tpot", "concurrency"]
    if "batch_size" in sub.columns:
        sub = sub.assign(_batch_sort=pd.to_numeric(sub["batch_size"], errors="coerce"))
        sort_cols.insert(1, "_batch_sort")
    sub = sub.sort_values(sort_cols)
    return sub.drop(columns=["_batch_sort"], errors="ignore")


def _emit_terminal_optimizer_curve_ascii(curve_df: pd.DataFrame, title_prefix: str) -> None:
    """Print throughput curves as terminal ASCII using plotext.

    Plotext is invoked via ``import plotext as plx`` and module-level calls such as
    ``plx.plot_size``, ``plx.theme``, ``plx.plot``, ``plx.build``, ``plx.clear_data``.
    The library does not provide a documented, caller-owned figure object comparable to
    matplotlib's ``Figure`` for isolating render state; subplot/matrix helpers still
    coordinate through plotext's internal active canvas.

    **Thread safety:** this function is **not** thread-safe. Do not call it concurrently
    from multiple threads or while other code uses plotext on overlapping timelines.
    The throughput optimizer CLI uses this only from a single-threaded path.

    The nested ``_one_chart`` runs twice **sequentially** (two charts per invocation);
    parallel ``_emit_terminal_optimizer_curve_ascii`` calls would still contend on
    plotext globals.
    """
    try:
        import plotext as plx
    except ImportError:
        logger.warning("plotext is not installed; skipping terminal curve plots.")
        return

    parallels = sorted(curve_df["parallel"].astype(str).unique())
    if not parallels:
        return

    # Sequential emission only; each call mutates plotext's shared canvas (see docstring).
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
            "Throughput vs concurrency",
            "concurrency",
            "Concurrency",
            "Throughput (token/s)",
            _sorted_sub_concurrency,
        )
        _one_chart(
            "Throughput vs TPOT",
            "tpot",
            "TPOT (ms)",
            "Throughput (token/s)",
            _sorted_sub_tpot,
        )
    except Exception:
        logger.exception("Terminal ASCII optimizer curves failed.")


def _prepare_curve_df(
    df: pd.DataFrame,
    ttft_limit: float | None,
    tpot_limit: float | None,
) -> pd.DataFrame:
    """Filter to SLO-feasible, non-OOM rows; keep every surviving sample (no averaging).

    Optimizer-produced aggregation frames typically omit OOM points already; if a frame
    includes optional memory columns, rows with explicitly non-positive headroom are dropped.
    """
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
        ttft_num = pd.to_numeric(work["ttft"], errors="coerce").fillna(float("inf"))
        work = work.loc[ttft_num <= float(ttft_limit)]

    if tpot_limit is not None:
        work = work.loc[
            pd.to_numeric(work["tpot"], errors="coerce").fillna(float("inf"))
            <= float(tpot_limit)
        ]

    for mem_col in ("memory_left_gb", "device_memory_available_gb"):
        if mem_col in work.columns:
            mem = pd.to_numeric(work[mem_col], errors="coerce")
            work = work.loc[mem.isna() | (mem > 0)]
            break

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
    """Merge per-job summary frames and print terminal curve plots."""
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
    """Print terminal ASCII charts (plotext): throughput vs concurrency and vs TPOT.

    Returns True when filtered data was non-empty and terminal emission was attempted.
    Does not write image files.

    Terminal rendering uses plotext module-level state and is not thread-safe; see
    ``_emit_terminal_optimizer_curve_ascii``.
    """
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

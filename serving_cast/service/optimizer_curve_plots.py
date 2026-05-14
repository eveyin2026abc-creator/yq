# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Optimizer curve plots: TPS vs concurrency, and TPS vs TPOT (per parallel case)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# PNG readability (matplotlib).
_FIGSIZE = (12, 7)
_SAVE_DPI = 180

# Terminal canvas (plotext).
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
    """Print throughput curves as terminal ASCII (plotext), aiconfigurator-style."""
    try:
        import plotext as plx
    except ImportError:
        logger.warning("plotext is not installed; skipping terminal curve plots.")
        return

    parallels = sorted(curve_df["parallel"].astype(str).unique())
    if not parallels:
        return

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


def _safe_filename_fragment(text: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w.\-]+", "_", text.strip()).strip("_")
    return s[-max_len:] if len(s) > max_len else s


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


def plot_concurrency_optimizer_curves(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    basename_prefix: str,
    ttft_limit: float | None = None,
    tpot_limit: float | None = None,
) -> tuple[str | None, str | None]:
    """Write two PNGs and print matching terminal ASCII charts (plotext).

    Plotext charts use one filtered row per marker (SLO + non-OOM), same as the PNGs.

    Returns:
        Tuple of output paths (tps_vs_concurrency_path, tps_vs_tpot_path),
        or (None, None) if nothing to plot.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = _safe_filename_fragment(basename_prefix)

    try:
        curve_df = _prepare_curve_df(df, ttft_limit, tpot_limit)
    except ValueError as exc:
        logger.warning("Skipping concurrency curve plots: %s", exc)
        return None, None

    if curve_df.empty:
        logger.warning(
            "Skipping concurrency curve plots: no rows after filtering "
            "(check TTFT/TPOT limits or optimizer output)."
        )
        return None, None

    parallels = sorted(curve_df["parallel"].astype(str).unique())
    if not parallels:
        return None, None

    cmap = plt.get_cmap("tab10")
    n_colors = getattr(cmap, "N", 10)

    def _plot_series(ax, x_series, y_series, idx: int, label: str) -> None:
        if hasattr(cmap, "colors"):
            color = cmap.colors[idx % len(cmap.colors)]
        else:
            color = cmap((idx % max(n_colors, 1)) / max(n_colors - 1, 1))
        ax.plot(
            x_series,
            y_series,
            marker="o",
            linestyle="-",
            linewidth=2.1,
            markersize=6.5,
            markeredgewidth=1.0,
            markeredgecolor="white",
            markerfacecolor=color,
            color=color,
            label=label,
        )

    def _finalize_axes(ax, *, title: str, x_label: str, y_label: str) -> None:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.tick_params(axis="both", which="major", labelsize=10, length=5)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.set_axisbelow(True)
        leg = ax.legend(
            fontsize=9,
            framealpha=0.93,
            loc="best",
            ncol=1 if len(parallels) <= 6 else 2,
            fancybox=True,
            shadow=False,
            edgecolor="0.78",
        )
        leg.get_frame().set_linewidth(0.8)

    fig1, ax1 = plt.subplots(figsize=_FIGSIZE, constrained_layout=True)
    fig2, ax2 = plt.subplots(figsize=_FIGSIZE, constrained_layout=True)

    for idx, parallel in enumerate(parallels):
        sub = _sorted_sub_concurrency(curve_df, parallel)
        if sub.empty:
            continue
        label = _parallel_label(parallel)
        _plot_series(ax1, sub["concurrency"], sub["token/s"], idx, label)

    for idx, parallel in enumerate(parallels):
        sub = _sorted_sub_tpot(curve_df, parallel)
        if sub.empty:
            continue
        label = _parallel_label(parallel)
        _plot_series(ax2, sub["tpot"], sub["token/s"], idx, label)

    _finalize_axes(
        ax1,
        title="Throughput vs concurrency (per parallel)",
        x_label="Concurrency",
        y_label="Throughput (token/s)",
    )
    _finalize_axes(
        ax2,
        title="Throughput vs TPOT (per parallel)",
        x_label="TPOT (ms)",
        y_label="Throughput (token/s)",
    )

    tps_path = out / f"{prefix}_tps_vs_concurrency.png"
    tps_tpot_path = out / f"{prefix}_tps_vs_tpot.png"

    fig1.savefig(tps_path, dpi=_SAVE_DPI)
    fig2.savefig(tps_tpot_path, dpi=_SAVE_DPI)
    plt.close(fig1)
    plt.close(fig2)

    logger.info("Wrote optimizer curve plots: %s , %s", tps_path, tps_tpot_path)
    title = str(basename_prefix).strip()[:160] or prefix
    _emit_terminal_optimizer_curve_ascii(curve_df, title_prefix=title)
    return str(tps_path), str(tps_tpot_path)

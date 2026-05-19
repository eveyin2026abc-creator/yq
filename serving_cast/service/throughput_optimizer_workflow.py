# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Workflow helpers for throughput optimizer CLI runs."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from serving_cast.service.optimizer_summary import (
    render_cross_device_comparison,
    render_cross_hardware_disagg_decode,
    render_cross_hardware_disagg_prefill,
    render_cross_hardware_pd_ratio,
    render_hardware_profile_comparison,
)


@dataclass
class MultiDeviceComparisonRows:
    aggregation: list[dict]
    pd_ratio: list[dict]
    disagg_prefill: list[dict]
    disagg_decode: list[dict]


def _first_non_empty_summary_df(results: list):
    for res in results:
        summary_df = res.get_summary_df()
        if summary_df is not None and not summary_df.empty:
            return summary_df
    return None


def _plot_single_device_optimizer_curves(
    results: list,
    args: argparse.Namespace,
    *,
    basename_prefix: str,
) -> None:
    """Dispatch terminal curve plotting for the active optimizer mode."""
    from serving_cast.service.optimizer_curve_plots import (
        plot_concurrency_curves_from_optimizer_summaries,
        plot_disagg_terminal_curves,
        plot_pd_ratio_terminal_curves,
    )

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


def run_multi_device_loop(
    args: argparse.Namespace,
    device_targets: list[str],
    *,
    plot_curves_allowed: bool,
    logger: logging.Logger,
) -> MultiDeviceComparisonRows:
    """Run ParallelRunner per device and collect cross-hardware rows."""
    from serving_cast.parallel_runner import ParallelRunner

    comparison_rows: list[dict] = []
    comparison_rows_pd: list[dict] = []
    comparison_rows_prefill: list[dict] = []
    comparison_rows_decode: list[dict] = []
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
            if multi_hw and args.disagg:
                pf_row = res.collect_disagg_prefill_row(profile_name)
                if pf_row:
                    comparison_rows_prefill.append(pf_row)
                dec_row = res.collect_disagg_decode_row(profile_name)
                if dec_row:
                    comparison_rows_decode.append(dec_row)
            elif multi_hw and args.enable_optimize_prefill_decode_ratio:
                pd_row = res.collect_pd_ratio_comparison_row(profile_name)
                if pd_row:
                    comparison_rows_pd.append(pd_row)
            elif multi_hw:
                row = res.collect_comparison_row(profile_name)
                if row:
                    comparison_rows.append(row)

        if plot_curves_allowed:
            _plot_single_device_optimizer_curves(
                results,
                args,
                basename_prefix=f"{profile_name}_{args.model_id}",
            )

    return MultiDeviceComparisonRows(
        aggregation=comparison_rows,
        pd_ratio=comparison_rows_pd,
        disagg_prefill=comparison_rows_prefill,
        disagg_decode=comparison_rows_decode,
    )


def render_cross_hardware_summary(
    args: argparse.Namespace,
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
        rendered_p = render_cross_hardware_disagg_prefill(rows.disagg_prefill)
        rendered_d = render_cross_hardware_disagg_decode(rows.disagg_decode)
        if rendered_p:
            print(rendered_p)
        if rendered_d:
            print(rendered_d)
        if not rows.disagg_prefill and not rows.disagg_decode:
            logger.warning(
                "No rows available for cross-hardware disaggregation comparison "
                "(all runs empty or limits omitted)."
            )
    elif args.enable_optimize_prefill_decode_ratio:
        rendered_pd = render_cross_hardware_pd_ratio(rows.pd_ratio)
        if rendered_pd:
            print(rendered_pd)
        elif not rows.pd_ratio:
            logger.warning(
                "No rows available for cross-hardware PD ratio comparison "
                "(all runs empty or filtered out)."
            )
    else:
        rendered = render_cross_device_comparison(rows.aggregation)
        if rendered:
            print(rendered)
        elif not rows.aggregation:
            logger.warning(
                "No rows available for cross-hardware comparison (all runs empty)."
            )

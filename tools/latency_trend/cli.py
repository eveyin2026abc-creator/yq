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

"""Command line for the batch-size latency trend report."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tools.latency_trend.report import render_html, report_payload
from tools.latency_trend.sweep import SweepConfig, run_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msmodeling inference latency-trend",
        description=(
            "Sweep batch size on one model and write an HTML report of network latency, "
            "the top operator latency curves, and where the dominant bound switches "
            "between memory and compute."
        ),
    )
    parser.add_argument("model_id", help="Hub id or local config directory")
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16,32,64",
        help="Comma-separated batch sizes (default: 1,2,4,8,16,32,64)",
    )
    parser.add_argument("--device", default="TEST_DEVICE")
    parser.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--query-length", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--quantize-linear",
        default="DISABLED",
        help="Linear quantization action (default DISABLED so the roofline is easier to read)",
    )
    parser.add_argument("--performance-model", default="analytic", choices=("analytic",))
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=0,
        help="Override layer count for a shorter trial. 0 keeps the model config.",
    )
    parser.add_argument(
        "--output",
        default="latency-trend.html",
        help="HTML report path. A .json file with the same stem is written beside it.",
    )
    return parser


def _parse_batch_sizes(text: str) -> list[int]:
    sizes = [int(part) for part in text.split(",") if part.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    return sizes


def config_from_args(args: argparse.Namespace) -> SweepConfig:
    if args.phase == "decode":
        query_length = 1 if args.query_length is None else args.query_length
        context_length = 4096 if args.context_length is None else args.context_length
    else:
        query_length = 2048 if args.query_length is None else args.query_length
        context_length = 0 if args.context_length is None else args.context_length
    return SweepConfig(
        model_id=args.model_id,
        batch_sizes=_parse_batch_sizes(args.batch_sizes),
        device=args.device,
        phase=args.phase,
        query_length=query_length,
        context_length=context_length,
        tp_size=args.tp_size,
        quantize_linear=args.quantize_linear,
        performance_model=args.performance_model,
        top_k=args.top_k,
        num_hidden_layers=args.num_hidden_layers,
    )


def write_report(report, output: str) -> tuple[Path, Path]:
    html_path = Path(output)
    html_path.write_text(render_html(report), encoding="utf-8")
    json_path = html_path.with_suffix(".json")
    json_path.write_text(json.dumps(report_payload(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return html_path, json_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = run_sweep(config_from_args(args))
    html_path, json_path = write_report(report, args.output)
    print(f"Wrote {html_path}")
    print(f"Wrote {json_path}")
    print(report_payload(report)["network_summary"])
    return 0

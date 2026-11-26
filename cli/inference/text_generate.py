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

import logging
import sys

from cli.logo import print_logo
from cli.registry.argparse_adapter import build_argparser, parse_module_args
from cli.registry.modules import get_spec
from cli.spec_cli import configure_std_logging
from tensor_cast import config, device_profiles  # noqa: F401
from tensor_cast.core.compilation_config import apply_compilation_config

from ..utils import (
    LOG_FORMAT,
    argv_has_option,
    draft_method,
    resolve_draft_block_and_acceptance,
)

# Supported performance model types
SUPPORTED_PERFORMANCE_MODELS = ["analytic", "calibrated", "profiling"]


def arg_parse(argv=None):
    """Parse CLI args and apply draft-spec block resolution.

    Parser is generated from the parameter registry; draft-spec G2/G3 checks
    run as registry validators inside parse_module_args (wants_provided=True
    — they see which flags were explicitly passed). Only the block-size
    resolution remains here: it MUTATES args (fills draft_block_size), which
    is post-parse business logic, not validation.
    """
    spec = get_spec("text_generate")
    parser = build_argparser(spec)
    raw_argv = list(argv) if argv is not None else None
    args = parse_module_args(spec, parser, argv)
    effective_argv = raw_argv if raw_argv is not None else sys.argv[1:]
    args.num_speculative_tokens_explicit = argv_has_option(effective_argv, "--num-speculative-tokens")
    resolve_draft_block_and_acceptance(args, argv=raw_argv)
    # Bridge --speculative-method mtp --num-speculative-tokens N onto the
    # legacy num_mtp_tokens field so ConfigResolver/MtpWrapper match
    # --num-mtp-tokens N.
    method = draft_method(args)
    if method is not None:
        n = int(getattr(args, "num_speculative_tokens", 0) or 0)
        if n == 0:
            parser.error(
                "--speculative-method is set but --num-speculative-tokens is explicitly 0 "
                "(disabled). Omit --speculative-method for a baseline run, or pass n >= 1."
            )
        if method == "mtp" and int(getattr(args, "num_mtp_tokens", 0) or 0) == 0:
            args.num_mtp_tokens = n
    if args.performance_model is None:
        args.performance_model = ["analytic"]
    if "calibrated" in args.performance_model and not args.analytic_calibration_profile:
        parser.error("--analytic-calibration-profile is required when using --performance-model calibrated")
    if args.export_empirical_metrics_file and "profiling" not in args.performance_model:
        parser.error("--export-empirical-metrics requires --performance-model profiling")
    if args.fusion_plugin and not args.compile:
        parser.error("--fusion-plugin requires --compile (else the fusion never fires)")
    # Validate fusion plugin paths eagerly so an invalid plugin is caught
    # before ModelRunner construction rather than silently falling back.
    if args.fusion_plugin:
        from tensor_cast.plugins.validator import validate_plugin

        for plugin_path in args.fusion_plugin:
            result = validate_plugin(plugin_path)
            if not result:
                parser.error(f"--fusion-plugin {plugin_path}: validation failed at {result.layer}: {result.detail}")
    _normalize_num_queries(args, parser)
    return args


def _normalize_num_queries(args, parser) -> None:
    """Keep ``num_queries`` as one integer and remember a multi-batch sweep."""
    raw = args.num_queries
    sizes = [int(item) for item in raw] if isinstance(raw, list) else [int(raw)]
    args.num_query_sweep = sizes
    args.num_queries = sizes[0]
    if len(sizes) > 1 and not getattr(args, "html_file", None):
        parser.error("--num-queries has more than one batch; pass --html-file FILE")
    if getattr(args, "html_port", None) and not getattr(args, "html_file", None):
        parser.error("--html-port requires --html-file")


def _scoped_output_path(path: str | None, batch_size: int, multiple: bool) -> str | None:
    """Keep one output path for a single run, and split files when several batches run."""
    if path is None or not multiple:
        return path
    from pathlib import Path

    file = Path(path)
    return str(file.with_name(f"{file.stem}.batch{batch_size}{file.suffix}"))


def _html_server(directory, host: str, port: int):
    """Serve one report directory. The caller decides how long it stays up."""
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    try:
        return ThreadingHTTPServer((host, int(port)), handler)
    except OSError as exc:
        raise SystemExit(f"Cannot listen on {host}:{port}: {exc}") from exc


def _serve_html_directory(directory, host: str, port: int, filename: str) -> None:
    """Serve ``directory`` until interrupted. The report URL uses ``filename``."""
    httpd = _html_server(directory, host, port)
    bound_host, bound_port = httpd.server_address[:2]
    shown_host = host if host != "0.0.0.0" else bound_host
    print(f"Serving {directory} at http://{shown_host}:{bound_port}/{filename}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Stopped HTML server")
    finally:
        httpd.server_close()


def _export_empirical_metrics(model_runner, path: str | None) -> None:
    """Write the existing profiling metrics JSON. An empty path exports nothing."""
    if not path:
        return
    from pathlib import Path

    from tensor_cast.performance_model.empirical import EmpiricalPerformanceModel
    from tensor_cast.performance_model.metrics_collector import MetricsCollector

    for pm in model_runner.perf_models:
        if isinstance(pm, EmpiricalPerformanceModel):
            collector = MetricsCollector()
            collector.collect_from_records(pm.op_records)
            collector.export_hit_miss_report(output_path=Path(path))
            return


def align_decode_query_length(args, logger=None) -> None:
    """Align decode ``--query-length`` to ``n + 1`` for MTP / DFlash / DSpark.

    Both MTP entries (legacy ``--num-mtp-tokens`` and
    ``--speculative-method mtp --num-speculative-tokens``) use the same window
    so decode simulation shapes stay identical.
    """
    method = draft_method(args)
    n = int(getattr(args, "num_speculative_tokens", 0) or 0)
    if method is None and int(getattr(args, "num_mtp_tokens", 0) or 0) > 0:
        n = int(args.num_mtp_tokens)
        method = "mtp"
    if not args.decode or n < 1:
        return
    decode_query_len = n + 1
    if int(args.query_length) == decode_query_len:
        return
    label = {"dspark": "DSpark", "dflash": "Dflash", "mtp": "MTP"}.get(method, method)
    if logger is not None:
        logger.warning(
            "%s decode sets --query-length to num_speculative_tokens+1 (%d); was %d",
            label,
            decode_query_len,
            args.query_length,
        )
    args.query_length = decode_query_len


def format_text_generate_command(arguments: list[str]) -> str:
    """Render the public text-generate command, one flag group per line."""
    import shlex

    quoted = [shlex.quote(arg) for arg in arguments]
    if not quoted:
        return "msmodeling inference text-generate"
    groups: list[list[str]] = []
    current: list[str] = []
    for token in quoted[1:]:
        if token.startswith("--") and current:
            groups.append(current)
            current = [token]
        else:
            current.append(token)
    if current:
        groups.append(current)
    lines = [f"msmodeling inference text-generate {quoted[0]}"]
    for group in groups:
        lines[-1] += " \\"
        lines.append("  " + " ".join(group))
    return "\n".join(lines)


def main():
    """
    Main function to parse arguments and run the inference simulation.

    Parser is generated from the declarative parameter registry
    (cli/registry/modules/text_generate.py) via build_argparser(); this
    module only holds the post-parse business logic.
    """
    args = arg_parse()
    command = format_text_generate_command(sys.argv[1:])
    if getattr(args, "html_file", None):
        print(command)
        print()
    print_logo()
    configure_std_logging(args, log_format=LOG_FORMAT)
    logger = logging.getLogger(__name__)

    if args.graph_log_url:
        config.compilation.debug.graph_log_url = args.graph_log_url
    apply_compilation_config(args.compilation_config)

    align_decode_query_length(args, logger=logger)

    # import here to make sure the logger level is set
    logger.info("Importing core modules...")
    from tensor_cast.core.input_generator import generate_inputs
    from tensor_cast.core.user_config import UserInputConfig

    logger.debug("Core modules imported")

    logger.info("Initializing user configuration...")
    user_input = UserInputConfig.from_args(args)
    logger.debug("User configuration initialized: %s", user_input)

    # Load fusion plugin(s) into the global tables before ModelRunner is built,
    # so Phase 3 picks them up at compile time. Additive hook (RFC §3.3a):
    # arg_parse() already validated each plugin path; load_plugin is
    # idempotent so calling it again here is harmless.
    if args.fusion_plugin:
        from tensor_cast.plugins.loader import load_plugin

        for plugin_path in args.fusion_plugin:
            load_plugin(plugin_path)

    logger.info("Initializing ModelRunner")
    from template_engine.runner_bridge import create_runner, wrap_registered_runner

    incompatible_reason = 'fusion_plugin' if args.fusion_plugin else ''
    model_runner = create_runner(
        user_input,
        enabled=args.compile and args.template_graph_optimization,
        incompatible_reason=incompatible_reason,
    )

    if args.compile:
        model_runner = wrap_registered_runner(
            model_runner,
            enabled=args.template_graph_optimization,
            incompatible_reason=incompatible_reason,
        )
    logger.info("ModelRunner initialization completed: %s", model_runner)

    sweep = [int(size) for size in (getattr(args, "num_query_sweep", None) or [args.num_queries])]
    html_file = getattr(args, "html_file", None)
    if html_file:
        from tools.latency_trend.cli import write_report
        from tools.latency_trend.report import report_payload
        from tools.latency_trend.sweep import sweep_loaded_runner

        # HTML is an extra report. The metrics table still prints, and existing
        # file outputs keep their own flags. Several batches must not overwrite
        # one chrome trace or one empirical JSON, so those paths gain a batch suffix.
        multiple = len(sweep) > 1
        original_trace = getattr(user_input, "chrome_trace", None)

        def _before_batch(batch_size: int) -> None:
            user_input.chrome_trace = _scoped_output_path(original_trace, batch_size, multiple)

        def _after_batch(batch_size: int, _metrics) -> None:
            empirical_path = _scoped_output_path(args.export_empirical_metrics_file, batch_size, multiple)
            _export_empirical_metrics(model_runner, empirical_path)

        logger.info("Running latency trend for batches %s", " ".join(str(size) for size in sweep))
        report = sweep_loaded_runner(
            model_runner,
            user_input,
            sweep,
            show_metrics=True,
            before_batch=_before_batch,
            after_batch=_after_batch,
        )
        user_input.chrome_trace = original_trace
        report.command = command
        html_path, json_path = write_report(report, html_file)
        print(f"Wrote {html_path}")
        print(f"Wrote {json_path}")
        print(report_payload(report)["network_summary"])
        if getattr(args, "html_port", None):
            _serve_html_directory(html_path.parent, args.html_bind, args.html_port, html_path.name)
        return 0

    logger.info("Running inference...")
    metrics = model_runner.run_inference(generate_inputs_func=generate_inputs)
    metrics.print_info()
    _export_empirical_metrics(model_runner, args.export_empirical_metrics_file)


if __name__ == "__main__":
    main()

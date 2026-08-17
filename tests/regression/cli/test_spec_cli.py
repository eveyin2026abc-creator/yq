"""Regression tests for MindStudio unified CLI spec (help, version, aliases)."""

from __future__ import annotations

import argparse
import re
import sys
from unittest.mock import patch

import pytest

from cli.inference import text_generate, throughput_optimizer
from cli.main import main
from cli.spec_cli import parse_args, reset_deprecation_warnings, to_kebab
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tests.helpers.cli_runner import run_cli_main, run_module_main


@pytest.fixture(autouse=True)
def _reset_deprecations() -> None:
    reset_deprecation_warnings()
    yield
    reset_deprecation_warnings()


def _capture_text_generate_args(argv: list[str]):
    captured: dict[str, argparse.Namespace] = {}

    def _capture(parser, args=None):
        captured["ns"] = parse_args(parser, args)
        raise SystemExit(0)

    with (
        patch("cli.inference.text_generate.spec_parse_args", side_effect=_capture),
        patch.object(sys, "argv", argv),
    ):
        try:
            text_generate.main()
        except SystemExit:
            pass
    return captured["ns"]


def test_top_level_help_has_required_sections() -> None:
    result = run_cli_main(main, ["--help"], prog="msmodeling")
    assert result.returncode == 0
    text = result.stdout
    assert "Description:" in text
    assert "Usage:" in text
    assert "Examples:" in text
    assert "--version" in text or "-V" in text


def test_top_level_version() -> None:
    result = run_cli_main(main, ["--version"], prog="msmodeling")
    assert result.returncode == 0
    assert "MindStudio" in result.stdout
    assert "msmodeling" in result.stdout
    assert "Mulan PSL v2" in result.stdout


def test_text_generate_help_hides_legacy_parallel_flags() -> None:
    result = run_module_main("cli.inference.text_generate", ["--help"])
    assert result.returncode == 0
    help_text = result.stdout
    assert "Description:" in help_text
    assert "Required arguments:" in help_text
    assert "Optional arguments:" in help_text
    assert "Examples:" in help_text
    assert "--tp-size" in help_text
    assert "--tensor-parallel-size" not in help_text
    assert "--chrome-trace-file" in help_text
    assert "--log-level" in help_text
    assert "-v" in help_text
    assert "-V" in help_text
    assert "(default: None)" not in help_text


def test_text_generate_tp_size_parses_without_deprecation(capsys: pytest.CaptureFixture[str]) -> None:
    ns = _capture_text_generate_args(
        [
            "text_generate",
            "Qwen/Qwen3-32B",
            "--num-queries",
            "1",
            "--query-length",
            "8",
            "--tp-size",
            "2",
        ]
    )
    assert ns.tp_size == 2
    assert "deprecated" not in capsys.readouterr().err


def test_text_generate_accepts_native_and_kebab_quant_choice() -> None:
    ns = _capture_text_generate_args(
        [
            "text_generate",
            "Qwen/Qwen3-32B",
            "--num-queries",
            "1",
            "--query-length",
            "8",
            "--quantize-linear-action",
            "W8A8_DYNAMIC",
        ]
    )
    assert ns.quantize_linear_action == QuantizeLinearAction.W8A8_DYNAMIC
    ns = _capture_text_generate_args(
        [
            "text_generate",
            "Qwen/Qwen3-32B",
            "--num-queries",
            "1",
            "--query-length",
            "8",
            "--quantize-linear-action",
            "w8a8-dynamic",
        ]
    )
    assert ns.quantize_linear_action == QuantizeLinearAction.W8A8_DYNAMIC


def test_throughput_optimizer_legacy_flags_parse() -> None:
    argv = [
        "throughput_optimizer",
        "--input-length=1",
        "--output-length=1",
        "Qwen/Qwen3-32B",
        "--num-devices",
        "8",
        "--tp-sizes",
        "1",
        "2",
        "--disagg",
        "--log-level",
        "debug",
    ]
    with patch.object(sys, "argv", argv):
        args = throughput_optimizer.arg_parse()
    assert args.tp_sizes == [1, 2]
    assert args.disagg is True
    assert args.log_level == "debug"


def test_throughput_optimizer_new_flags_parse() -> None:
    argv = [
        "throughput_optimizer",
        "--input-length=1",
        "--output-length=1",
        "Qwen/Qwen3-32B",
        "--num-devices",
        "8",
        "--tp-sizes",
        "1",
        "2",
        "--disagg",
        "--verbose",
    ]
    with patch.object(sys, "argv", argv):
        args = throughput_optimizer.arg_parse()
    assert args.tp_sizes == [1, 2]
    assert args.disagg is True
    assert args.log_level == "debug"


def test_optix_help_hides_snake_case_and_multichar_short() -> None:
    pytest.importorskip("pydantic_settings")
    result = run_module_main("optix", ["--help"])
    assert result.returncode == 0
    help_text = result.stdout
    assert "--load-breakpoint" in help_text
    assert "--benchmark-policy" in help_text
    assert "ais_bench" in help_text
    assert "vllm_benchmark" in help_text
    assert "ais-bench" not in help_text
    assert "vllm-benchmark" not in help_text
    assert "--load_breakpoint" not in help_text
    assert "-lb" not in help_text
    assert "--version" in help_text
    assert "Description:" in help_text


def test_optix_legacy_load_breakpoint_still_accepted(capsys: pytest.CaptureFixture[str]) -> None:
    parser = argparse.ArgumentParser()
    from cli.spec_cli import add_option

    add_option(
        parser,
        "--load-breakpoint",
        dest="load_breakpoint",
        action="store_true",
        aliases=("--load_breakpoint", "-lb"),
    )
    ns = parse_args(parser, ["--load_breakpoint"])
    assert ns.load_breakpoint is True
    assert "deprecated" in capsys.readouterr().err


def test_to_kebab_enum_values() -> None:
    assert to_kebab("W8A8_DYNAMIC") == "w8a8-dynamic"
    assert to_kebab("block_sparse_attention") == "block-sparse-attention"
    assert to_kebab("linear_exponential") == "linear-exponential"


def test_log_level_verbose_and_quiet_resolution() -> None:
    from cli.spec_cli import resolve_log_level

    args = argparse.Namespace(log_level=None, verbose=True, debug=False, quiet=True)
    assert resolve_log_level(args) == "debug"
    args = argparse.Namespace(log_level="warning", verbose=True, debug=False, quiet=True)
    assert resolve_log_level(args) == "warning"
    args = argparse.Namespace(log_level=None, verbose=False, debug=False, quiet=True)
    assert resolve_log_level(args) == "error"
    args = argparse.Namespace(log_level=None, verbose=False, debug=False, quiet=False)
    assert resolve_log_level(args) == "info"


def _assert_help_meets_spec(help_text: str) -> None:
    assert "Description:" in help_text
    assert "Usage:" in help_text
    assert "Examples:" in help_text
    assert "(default: None)" not in help_text
    assert re.search(r"--[A-Za-z0-9]*_[A-Za-z0-9_]+", help_text) is None
    for line in help_text.splitlines():
        if re.match(r"^  -[A-Za-z0-9]{2}", line):
            raise AssertionError(f"multi-character short option in help: {line!r}")


def test_text_generate_help_uses_native_enum_defaults() -> None:
    result = run_module_main("cli.inference.text_generate", ["--help"])
    assert result.returncode == 0
    _assert_help_meets_spec(result.stdout)
    assert "[default: W8A8_DYNAMIC]" in result.stdout
    assert "--chrome-trace-file" in result.stdout
    assert "--chrome-trace" not in result.stdout.replace("--chrome-trace-file", "")


def test_throughput_optimizer_help_hides_legacy_flags() -> None:
    result = run_module_main("cli.inference.throughput_optimizer", ["--help"])
    assert result.returncode == 0
    _assert_help_meets_spec(result.stdout)
    assert "--tp-sizes" in result.stdout
    assert "--tensor-parallel-sizes" not in result.stdout
    assert "--disagg" in result.stdout
    assert "--disaggregation" not in result.stdout
    assert "-j," in result.stdout
    assert "--jobs" in result.stdout


def test_video_generate_help_meets_spec() -> None:
    result = run_module_main("cli.inference.video_generate", ["--help"])
    assert result.returncode == 0
    _assert_help_meets_spec(result.stdout)
    assert "--ulysses-size" in result.stdout
    assert "--ulysses-parallel-size" not in result.stdout
    assert "--num-devices" in result.stdout
    assert "--world-size" not in result.stdout


def test_model_adapter_subcommand_help_meets_spec() -> None:
    for argv in (["--help"], ["doctor", "--help"], ["verify", "--help"], ["export-evidence", "--help"]):
        result = run_module_main("cli.inference.model_adapter", argv)
        assert result.returncode == 0, argv
        _assert_help_meets_spec(result.stdout)
        if argv[0] != "--help":
            assert "-o," in result.stdout
            assert "--output-file" in result.stdout
            assert re.search(r"--output[^-a-z]", result.stdout) is None


def test_top_level_help_lists_commands() -> None:
    result = run_cli_main(main, ["--help"], prog="msmodeling")
    assert result.returncode == 0
    _assert_help_meets_spec(result.stdout)
    assert "Commands:" in result.stdout
    assert "inference" in result.stdout
    assert "optix" in result.stdout

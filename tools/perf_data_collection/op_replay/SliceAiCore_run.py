"""Replay SliceAiCore cases through the same torch_npu Slice API."""

from __future__ import annotations

try:
    from .Slice_run import build_slice_case, format_success, run_case
    from .replay_framework import OpReplay
except ImportError:
    from Slice_run import build_slice_case, format_success, run_case
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    return build_slice_case(op, row)


op = OpReplay(
    kernel_type="SliceAiCore",
    api_path="torch_npu.npu_slice",
    description="Replay SliceAiCore.csv through torch_npu.npu_slice on Ascend NPU.",
    usage_examples=[
        "python tools/perf_data_collection/op_replay/SliceAiCore_run.py --database-path /path/to/database"
    ],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    build_case=build_case,
    run_case=run_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()

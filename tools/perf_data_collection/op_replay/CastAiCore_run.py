"""Replay CastAiCore cases from the performance database on Ascend NPU."""

from __future__ import annotations

try:
    from .Cast_run import build_cast_case, run_case
    from .replay_framework import OpReplay
except ImportError:
    from Cast_run import build_cast_case, run_case
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    return build_cast_case(op, row)


op = OpReplay(
    kernel_type="CastAiCore",
    description="Replay CastAiCore.csv dtype conversion cases on Ascend NPU.",
    usage_examples=["python tools/perf_data_collection/op_replay/CastAiCore_run.py --database-path /path/to/database"],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    input_count=1,
    build_case=build_case,
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()

"""Replay LayerNormV3 rows with the three-output native layer-norm API."""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    inputs = op.build_inputs(row)
    if len(inputs) != 3:
        raise ValueError("LayerNormV3 expects input, weight, and bias tensors")
    input_tensor, weight, bias = inputs
    if len(input_tensor.shape) < 1 or tuple(weight.shape) != tuple(bias.shape):
        raise ValueError("LayerNormV3 weight and bias shapes must match")
    normalized_shape = tuple(weight.shape)
    if tuple(input_tensor.shape[-len(normalized_shape) :]) != normalized_shape:
        raise ValueError("LayerNormV3 normalized shape must match the input tail")
    return {
        "inputs": inputs,
        "kwargs": {
            "normalized_shape": normalized_shape,
            "eps": 1e-5,
        },
        "api": op.resolve_api(),
    }


def run_case(case):
    input_tensor, weight, bias = case["inputs"]
    return case["api"](
        input_tensor,
        case["kwargs"]["normalized_shape"],
        weight,
        bias,
        case["kwargs"]["eps"],
    )


def format_success(csv_path, row_index: int, row: dict[str, str], _case, result) -> str:
    output, mean, rstd = result
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(output.shape)} "
        f"mean={tuple(mean.shape)} rstd={tuple(rstd.shape)}"
    )


op = OpReplay(
    kernel_type="LayerNormV3",
    api_path="torch.native_layer_norm",
    description=(
        "Run LayerNormV3 workload replay on Ascend NPU.\n"
        "The three input tensors are replayed through torch.native_layer_norm "
        "so the output, mean, and reciprocal-standard-deviation contracts "
        "match the profiling CSV."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/LayerNormV3_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.18.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    input_count=3,
    build_case=build_case,
    run_case=run_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()

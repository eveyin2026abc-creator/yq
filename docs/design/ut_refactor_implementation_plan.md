# L1 / L2 / L3 migration register

This document registers the current test suite. It does not move, merge, or delete any test, and it does not change a pytest node id.

The register was collected from commit `2522c830` with:

```bash
python -m pytest tests -o addopts= --collect-only -q
```

Clearing `addopts` keeps `nightly`, `npu`, and `network` nodes in the register. The collection result is **9154** node ids across **395** files. The same rows are in [`l1_l2_l3_test_inventory.xlsx`](l1_l2_l3_test_inventory.xlsx). One row is one collected node id. Re-running collection on this commit must still report 9154; a different count means the workbook is stale.

## What a row means

| Column | Meaning |
|--------|---------|
| `node_id` | Pytest node id. This is the primary key. |
| `path` | Test module, the part of `node_id` before `::`. |
| `target_layer` | Proposed responsibility layer. `待人工确认` means no rule below matched. |
| `target_module` | First component directory under `tests/regression/`, or `smoke` / `benchmark/...` / `helpers`. |
| `model_identity` | `model_id` literals found in that file, or `无模型`. This is file-level evidence, not a per-node parameter. |
| `markers` | Markers attached to that node. |
| `schedule` | Where this node runs today. |
| `migration` | `保持` or the later migration step. No code in this change performs it. |
| `evidence` | The rule that selected the layer. |
| `build_signature` | Filled only for proposed L1 rows. |
| `assertion_overlap` | `numeric`, `diagnostics`, or `未判定` for proposed L1 rows. |

`schedule` is taken from the path and markers:

| Condition | `schedule` |
|-----------|------------|
| Path under `tests/benchmark/` | `benchmark` (749) |
| `@pytest.mark.npu` | `excluded-npu` (8) |
| `@pytest.mark.network` | `nightly-network` (24) |
| `@pytest.mark.nightly` | `nightly` (219) |
| Everything else | `incremental` (8154) |

## Layer rules

These rules are applied in order. A later rule does not override an earlier one.

| Order | Rule | `target_layer` | `migration` | Rows |
|------:|------|----------------|-------------|-----:|
| 1 | Path under `tests/smoke/` | `smoke` | `保持` | 123 |
| 2 | Path under `tests/benchmark/models/` | `L1` | `保持` | 39 |
| 3 | Other path under `tests/benchmark/` | `benchmark-op` | `保持` | 710 |
| 4 | Path contains `model_diagnostics/e2e/` | `L1` | `待归并到 L1` | 149 |
| 5 | Path under `tests/regression/scripts/` or `tests/helpers/tests/` | `L3` | `保持` | 811 |
| 6 | File text contains `ModelRunner`, `build_model`, or both | `L1` | `待归并到 L1` | 900 + 448 + 260 |
| 7 | None of the above | `待人工确认` | `待人工确认` | 5714 |

Proposed L1 is **1796** rows: 39 already live under `tests/benchmark/models/`, and **1757** are regression nodes still to be merged onto `tests/helpers/l1_scenario.py`. `assertion_overlap` is `numeric` for benchmark model cases, `diagnostics` for `model_diagnostics/e2e`, and `未判定` for the other L1 rows. This register does not decide whether those assertions share one build.

`待人工确认` is intentional. Those nodes did not show a real model build, and they are not toolchain or helper contract tests. They are not labeled L2 or L3 here.

For every proposed L1 row, `build_signature` is the same pending contract already implemented in `tests/helpers/l1_scenario.py`: every `UserInputConfig` field except `RUN_FIELDS` and `OBSERVABILITY_FIELDS`, plus `compilation_config`. The workbook does not instantiate a signature for each node.

## Approvals required before migration code

Two decisions are still open. Neither is made by this register.

1. Confirm the `L1BuildSignature` field split and the runner rebind contract in `tests/helpers/l1_scenario.py`.
2. Approve the model-family matrix for numeric, structural, and capability assertions, and the incremental / regression / nightly time budget (about 180 / 300 / 3000 seconds).

Until both are approved, later pull requests must not move node ids. The next code change, after that approval, is the Qwen L1 merge only.

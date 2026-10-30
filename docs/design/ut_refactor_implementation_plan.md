# L1 / L2 / L3 migration implementation and acceptance record

This document is the executable migration contract and final acceptance record. The inventory workbook is generated from the current checkout by `scripts/helpers/export_test_inventory.py`; it is routing evidence, while code, collect results, build/forward counters, assertion manifests and `test_map` watcher checks are completion evidence.

The current register was collected on the refactor branch with:

```bash
python -m pytest tests -o addopts= --collect-only -q
```

Clearing `addopts` keeps `nightly`, `npu`, and `network` nodes in the register. The 2026-10-29 refresh collected **10,279 node ids**. The same rows are in [`l1_l2_l3_test_inventory.xlsx`](l1_l2_l3_test_inventory.xlsx), one row per collected node id. Re-run the exporter whenever collection changes; a count mismatch means the workbook is stale. The historical 6,943/7,626/9,154 counts are not acceptance baselines.

## Final implementation status

- `L1BuildSignature`, `L1RunSignature`, `L1RunArtifact`, reset/rebind isolation and counted execution are implemented in `tests/helpers/l1_scenario.py`.
- Qwen, dense, MoE and multimodal family scenarios execute real vendored one-layer configurations. The registry includes Qwen, DeepSeek V3/V3.2/V4, GLM, MiniMax, MiMo, Ling and Kimi families rather than treating Qwen/DeepSeek as the complete model set.
- Numeric, structural and capability assertions are protected by `tests/.ci/assertion_manifest.csv`; active assertion loss requires a signed waiver. Scenario budgets count `(build, run)` scenarios, not pytest case count.
- Canonical L2 artifacts and L3 pure contracts are implemented. The L2 AST guard scans every `tests/regression/*/module/**/*.py` file plus the canonical helper and runs from `run_ci_gate.sh`.
- Old-path watcher conservation is enforced through `l1_watchers.csv` and the external `test_map`. Jaccard redundancy candidates are advisory in test-map sync and nightly/Feishu output; they never delete a test automatically.
- Nightly reports the unchanged line/branch leadership gate plus core coverage and `cov_helper` contribution separately. `cov_helper` is an L3 label, not a fourth responsibility layer.
- The physical hierarchy is component first (`cli`, `tensor_cast`, `model_diagnostics`, etc.), then responsibility (`model`, `module`, `unit`). Existing out-of-scope suites retain an explicit maintenance boundary in the workbook and `out_of_scope_boundaries.csv`.

## What a row means

| Column | Meaning |
|--------|---------|
| `node_id` | Pytest node id. This is the primary key. |
| `当前路径` | Test module, the part of `node_id` before `::`. |
| `目标责任层` | Executable responsibility layer or registered maintenance boundary. |
| `target_module` | First component directory under `tests/regression/`, or `smoke` / `benchmark/...` / `helpers`. |
| `model_identity` | `model_id` literals found in that file, or `无模型`. This is file-level evidence, not a per-node parameter. |
| `markers` | Markers attached to that node. |
| `schedule` | Where this node runs today. |
| `迁移动作` | Keep, completed migration, or maintenance-boundary action. |
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

## Approval and rollback rules

The build-signature split, runner rebind contract, family matrix and scenario budget are now encoded in executable tests and manifests. Future changes to those contracts require review of the affected manifest rows. Assertion loss, signature collision, state leakage, watcher loss or an unexplained budget regression blocks the gate and requires rollback or an explicit signed waiver.

## Code already on this branch

Smoke guards in `tests/smoke/test_inference_smoke.py`, `test_fusion_passes_smoke.py`, `test_model_runner_compile_smoke.py`, and `test_compile_remote_models_smoke.py` use the same entry, as do the text benchmark cases in `tests/benchmark/models/test_model_regression.py` and the agg/disagg optimizer setups. Node ids are unchanged.

`tests/helpers/l1_case_matrix.py` lists every vendored config and the reuse probes that do not need a forward pass. `tests/helpers/l1_diagnostics.py` turns a diagnostics profile into the same user input. `tests/helpers/l1_field_roles.py` records why each `UserInputConfig` field is a build, run, or observability field. `tests/helpers/l2_catalog.py` is the accepted and rejected sample list for the seven L2 builders. `tests/helpers/construction_census.py` lists remaining direct constructions, and `tests/helpers/l1_migration_backlog.py` says which of those stay and which wait. `tests/helpers/inventory_rules.py` is the same layer order as the register. Diagnostics e2e captures go through `capture_profile_with_executor`. Session model caches and the `UserInputConfig` build sites go through `model_from_user_input`. Layer tests that already hold a `ModelConfig` go through `layer_transformer`. Serving cases go through `serving_model_runner`. None of these move node ids. The product function `capture_artifact_for_profile` is unchanged. Pipeline-builder tests still call `build_model` because they test that function.

`isolated_l1_executor()` is the one-case entry. `test_text_generate.py`, the Kimi K2.5 / K3 cases, GLM5-Next compile cases, the VL compile case, the matmul-allreduce and sequence-parallel pass cases, the DFC model runs, the DFlash varlen run, and the serving optimizer factory now build through it. Pass flags that the executor would otherwise clear are passed back in `compilation_config`. Node ids are unchanged.

`tests/regression/tensor_cast/model/test_text_generate.py` now runs its cases through `L1ScenarioExecutor`. Metric-only cases and model inspections use the class executor (`_run_shared`, `_shared_runner`). Compile cases and the expected-failure device case use `_isolated_executor`, so a compiled runner is not kept across `torch.compiler.reset()`. Node ids are unchanged.

`tests/helpers/l2_artifacts.py` adds the seven synthetic L2 builders: quant, parallel, FX, runtime event, diagnostics, replay, and microbench. They reject model identity. `tests/helpers/l3_contracts.py` adds pure exit-code, exception, and default contracts. `tests/helpers/layer_guard.py` scans L2 call sites for real model construction; `check-l2-layers` is wired into `run_ci_gate.sh` and currently covers 77 files with zero violations. Diagnostics e2e cases build through `capture_profile_with_executor`; product-entry integration tests retain direct `capture_artifact_for_profile` calls because that function is their subject under test.

`tests/helpers/l1_qwen.py` and `tests/regression/tensor_cast/model/test_l1_qwen.py` cover the reuse contract on Qwen3-32B with `num_hidden_layers_override=1`:

- the same scenario builds once and forwards once
- a different `query_len` reuses that build
- a different quantization is a second build

`tests/helpers/l1_families.py` and `tests/regression/tensor_cast/model/test_l1_families.py` apply that same contract to the vendored one-layer configs for Qwen3 dense, Qwen3 MoE, Qwen3-VL, Qwen3-VL-MoE, Qwen3.5, Qwen3.5-MoE, Qwen3-Next, DeepSeek-V3, DeepSeek-V3.2, DeepSeek-V4-Flash, GLM-4.7, GLM-4.1V, GLM-4.5V, GLM-5.1, MiniMax-M2, MiMo-V2-Flash, Ling-flash-2.0, and Kimi-K2.5. Kimi-K2-Thinking stays on the network-marked diagnostics cases and is not in this offline set.

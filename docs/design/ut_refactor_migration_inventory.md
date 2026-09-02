# UT refactor migration inventory

Working list for the TensorCast / CLI / benchmark / smoke wave. Rules and deletion gate:
[ut_refactor.md](./ut_refactor.md), daily placement: [tests/README.md](../../tests/README.md).

> **2026-09-02 routing override:** L1 now owns every real-model case (precision JSON,
> diagnostics E2E, real `model_id` / `ModelRunner` / model build-forward-compile).
> L2 is no-model integration routed by one of seven canonical synthetic artifacts;
> pure parser/validation/exit-code/helper cases are L3. The older PR2/PR3 rows below
> record migration history and their `module = L2` labels are not the target state.
> The refreshed static reclassification is
> [l2_seven_artifact_inventory.xlsx](./l2_seven_artifact_inventory.xlsx). Its
> `v2三线重叠35条上收` sheet records the 35 former L2 candidates now routed to L1.

PR2 created `tests/regression/{tensor_cast,cli}/{unit,module}/` and moved the first-wave whole-file L3 cases under the former definition. The new target reuses those physical directories but changes responsibility: `model/` and diagnostics `e2e/` are L1; `module/` must become no-model L2; `unit/` is L3. Later PRs copy `current_node_id` → `target_node_id` after each remaining move/split. File moves change pytest node ids; refresh the external `test_map` after merge (nightly).

## How to read a row

| Column | Meaning |
|--------|---------|
| Current path | File still on disk |
| Layer | L1 / L2 / L3 / smoke / operator-DB / keep-layout |
| Action | `keep` / `move` / `split` / `share-fixture` / `review` |
| Target | Intended or current path under `model/` / `unit/` / `module/` |
| Risk | `test_map` sole watcher, session cache, compile, platform |

Durations are Windows serial, 2026-08-29, default markers `not npu and not nightly and not network`. They rank work; they are not Linux/xdist CI times.

## Measured slow nodes (≥20 s)

| Duration | Current node id | Layer | Action |
|----------|-----------------|-------|--------|
| 77.51 s | `tests/benchmark/models/test_model_regression.py::TestPerformanceRegression::test_performance_regression_04_kimi_k2_5_16k_decode` | L1 | `keep` — do not delete; judge baseline only on standard runners |
| 74.79 s | `tests/regression/tensor_cast/test_video_generate.py::TestVideoGeneration::test_video_inference_with_raw_tencent_hunyuanvideo15_t2v_selector` | L2 | `review` — keep cheap smoke; consider `@pytest.mark.nightly` for the full path |
| 72.68 s | `…::test_performance_regression_02_kimi_k2_5_1080p_decode` | L1 | `keep` |
| 65.49 s | `…::test_performance_regression_05_kimi_k2_5_16k_prefill` | L1 | `keep` |
| 63.25 s | `tests/smoke/test_throughput_optimizer_smoke.py::TestThroughputOptimizerSmoke::test_vl_moe_aggregation_compile_smoke` | smoke | `review` — too expensive for smoke; shrink or mark nightly + tiny guard |
| 63.10 s | `…::test_performance_regression_03_kimi_k2_5_1080p_prefill` | L1 | `keep` |
| 41.68 s | `…::test_prefix_cache_hit_rate_aggregation_valid` | smoke | `review` vs CLI regression same scenario |
| 37.05 s | `…::test_deepseek_pd_ratio_mode` | smoke | `review` — keep one cheap point |
| 27.84 s | `tests/regression/scripts/helpers/common/test_build_test_map.py::test_collect_allowed_node_ids_includes_build_helper_regression_tests` | keep-layout | `review` — avoid full collect inside the test |
| 22.21 s | `tests/regression/cli/test_throughput_optimizer.py::TestThroughputOptimizer::test_aggregation_functionality_with_output_validation` | L2 | `share-fixture` with smoke; then `split` parse vs run |
| 22.07 s | `…::test_performance_regression_01_GLM_4_7_decode` | L1 | `keep` |
| 21.18 s | `…::test_performance_regression_06_kimi_k2_5_3_5k_decode` | L1 | `keep` |
| 20.62 s | `tests/regression/tensor_cast/test_parallel_linear.py::ParallelLinearTestCase::test_deepseek_with_tp_and_dp_0_deepseek_ai_DeepSeek_V3_1` | L2 | `share-fixture` / consider nightly |
| 20.61 s | `…::test_vl_disagg_prefill_smoke` | smoke | `review` |
| 20.51 s setup | `tests/regression/tensor_cast/test_input_generator.py::test_dsa_indexer_cache_dtype_follows_attention_quant_config` | L3+L2 mix | `split` setup vs assertion |
| 20.43 s | `…::test_vl_model_image_args` | smoke | `review` |
| 20.38 s setup | `…::test_dsa_indexer_cache_dtype_uses_fp8_when_attention_quant_is_fp8` | L3+L2 mix | `split` |
| 20.16 s | `tests/smoke/test_fusion_passes_smoke.py::test_dfc_dispatch_ffn_combine_smoke` | smoke | `keep` if it is the cheap DFC guard; do not duplicate L2 compile matrix |

Also slow (7–16 s), same action class: several `TestTextGenerate` nodes, CLI disagg validation, `test_sp_throughput_optimizer_row_embedding_tp_*`, `test_runtime.py::…::test_deepseek_0_*`.

## Node-level splits (must not `move` the whole file)

### `tests/regression/tensor_cast/test_runtime.py`

| Current node (examples) | Layer | Target after later PR | Action |
|-------------------------|-------|----------------------|--------|
| `PerfAnalysisTestCase::test_bound_analyzer_*` | L3 | `unit/test_runtime_perf_model.py` | `split` |
| `PerfAnalysisTestCase::test_runtime_breakdown_*` | L3 | same | `split` |
| `PerfAnalysisTestCase::test_deepseek_*` / `test_model_*` | L1 candidate | `model/test_text_model_contract.py` | `merge into the all-model matrix` |
| `PerfAnalysisNightlyTestCase::*` | L2 + nightly | `module/test_perf_analysis.py` | `keep` marker |

### `tests/regression/tensor_cast/test_text_generate.py`

| Current node (examples) | Layer | Target | Action |
|-------------------------|-------|--------|--------|
| `TestUserInputConfigPrintInfo::*` | L3 | merge into `unit/` + existing `test_user_config.py` after assertion compare | `review` then `split` |
| `TestModelRunnerMetricsPrintInfo::*` / `TestAggregateRuntimeEvents::*` | L3 | `unit/test_model_runner_unit.py` | `split` |
| `TestTextGenerate::test_*` (full inference) | L2 | `module/test_text_generate.py` | `split` + `share-fixture` (`make_user_input_config`) |
| `TestTextGenerateNightly::*` | L2 + nightly | same module file | `keep` marker + smoke guard |

Example **not** to delete yet: `test_main_invalid_prefix_cache_hit_rate_exits_with_code_2` vs CLI `test_prefix_cache_hit_rate_aggregation_valid` share `check_prefix_cache_hit_rate` (Jaccard 1 on one symbol) but assert different CLI vs optimizer outcomes.

### `tests/regression/tensor_cast/test_video_generate.py`

| Current node | Layer | Action |
|--------------|-------|--------|
| `TestVideoGeneration::test_video_inference_with_raw_tencent_hunyuanvideo15_t2v_selector` | L2 | `review` nightly; keep smoke reachability |

### Interpolation overlap (review only)

These pairs share one entry symbol (`InterpolatingDataSource.lookup` or `CandidateGroup::_filter_candidates`). Compare full assertions before any delete.

| Node A | Node B | Action |
|--------|--------|--------|
| `unit/test_interpolating_data_source.py::test_partial_falls_through_to_interpolation` | `test_profiling_interpolation_phase1.py::test_wrapper_interpolation_records_partial_fallback_source` | `review` — fallback vs source tag |
| `unit/test_interpolating_data_source.py::test_partial_returns_none_when_interpolation_fails` | same phase1 wrapper test | `review` |
| `test_profiling_interpolation_phase1.py::test_candidate_filter_rejects_missing_non_selected_axis` | `test_specialized_operator_interpolation.py::test_sparse_attention_interpolates_arbitrary_single_request_prefill` | `review` |
| `test_profiling_interpolation_phase1.py::test_candidate_group_1d_rejects_zero_latency_interpolation` | same specialized sparse-attention test | `review` |

## File-level inventory — TensorCast

Target prefix: `tests/regression/tensor_cast/{model,unit,module}/`. PR2 moved the first-wave L3 files into `unit/` (no nested conftest); the all-model audit adds `model/`.

| Current path | Layer | Action | Target |
|--------------|-------|--------|--------|
| `unit/test_ops.py`, `unit/test_dtype.py`, `unit/test_layers.py` | L3 | `keep` (PR2 moved) | `unit/` |
| `unit/test_pattern_match.py`, `unit/test_sp_pass_unit.py` | L3 | `keep` (PR2 moved; pattern_match now imports `test_common`/`conftest` absolutely) | `unit/` |
| `unit/test_user_config.py`, `unit/test_quantization_config_create.py` | L3 | `keep` (PR2 moved) | `unit/` |
| `unit/test_config_resolver.py`, `unit/test_auto_model_config_loader.py` | L3 | `keep` (PR2 moved) | `unit/` |
| `unit/test_interpolation_math.py`, `unit/test_comm_analytic.py` | L3 | `keep` (PR2 moved) | `unit/` |
| `unit/test_graph_extractor.py`, `unit/test_memory_tracker.py` | L3 | `keep` (PR2 moved) | `unit/` |
| `unit/test_model_hub.py`, `unit/test_image_dispatch.py` | L3 | `keep` (PR2 moved) | `unit/` |
| `unit/test_dfc_default.py`, `unit/test_plugin_validator.py`, `unit/test_l3_real.py` | L3 | `keep` (PR2 moved) | `unit/` |
| `unit/test_helpers_usage.py`, `unit/test_parameterized_pytest_param_compat.py` | L3 | `keep` (PR2 moved) | `unit/` |
| `unit/test_device.py`, `unit/test_transformers_utils.py` | L3 | `keep` (PR2 moved) | `unit/` |
| `unit/test_model_source_security.py` | L3 | `keep` (PR2 moved; Windows symlink failures are platform, not delete) | `unit/` |
| `test_common.py` | L3 helper | `keep` next to component `conftest.py` | later |
| `unit/test_interpolating_data_source.py` | L3 | `keep` (PR2 moved; fixture path uses `parents[3]`); later absorb generic interpolation from phase1/mla | `unit/test_interpolating_data_source.py` |
| `unit/test_empirical.py` | L3 | `keep`; database sibling is `unit/test_empirical_database.py` | `unit/` |
| `unit/test_profiling_interpolation_phase1.py` | L3 (oversized) | `keep` this wave; later split interpolation files | `unit/` |
| `unit/test_specialized_operator_interpolation.py` | L3 | `keep` this wave | `unit/` |
| `unit/test_input_generator.py` | L3 | `moved` (helpers + DSA cache dtype stay together) | `unit/` |
| `module/test_sequence_parallel_pass.py` | L2 compile / optimizer | `moved`; FX rewriter regressions extracted | `module/` + `unit/test_sp_pass_regression.py` |
| `module/test_dfc_pass.py` | L2 nightly compile | `moved`; FX unit class extracted | `module/` + `unit/test_dfc_pass.py` |
| `module/test_runtime.py` | L2 | `moved`; BoundAnalyzer / breakdown extracted | `module/` + `unit/test_runtime_perf_model.py` |
| `module/test_text_generate.py` | L2 | `moved`; print_info collectors live in `unit/test_model_runner_unit.py` | `module/` |
| `module/test_mtp.py`, `module/test_mtp_ep.py` | L2 | `moved`; resolve/rotary helpers extracted | `module/` + `unit/test_mtp_helpers.py` |
| `model/test_model_load.py` | L1 | `moved`; nightly compile matrix is `module/test_model_load_deep.py` | `model/` |
| `module/test_repetition.py` | L2 | `moved` | `module/` |
| `module/test_gmm_pass.py`, `module/test_swiglu_fusion_pass.py` | L2 | `moved` | `module/` |
| `module/test_parallel_linear.py`, `module/test_parallel_moe.py` | L2 | `moved`; shard helpers extracted | `module/` + `unit/test_parallel_linear_helpers.py` |
| `unit/test_parallel_embedding.py` | L3 | `moved` | `unit/` |
| `module/test_matmul_allreduce_pass.py`, `module/test_merge_linear_pass.py` | L2 | `moved` | `module/` |
| `unit/test_multistream_pass.py`, `unit/test_shape_cat_passes.py` | L3 | `moved` | `unit/` |
| `module/test_quant_linear.py`, `module/test_quant_attention.py` | L2 compile | `moved` this wave; helper-vs-compile node split later | `module/` |
| `unit/test_quant_config.py` | L3 | `moved` | `unit/` |
| `module/test_vl_compile.py` | L2 | `moved` | `module/` |
| `module/test_video_generate.py` | L2 | `moved` + nightly review still open | `module/` |
| `module/test_image_generation_*_e2e.py` | L2 | `moved` | `module/` |
| `unit/test_image_generation_flux1_dev.py`, `unit/test_image_generation_qwen_image_edit.py` | L3 fixture/dispatch | `moved` | `unit/` |
| `unit/model_adaptation/test_{glm5,bailing_v3,minimax_m2,deepseek_v4,kimi_k25,kimi_k3}.py` | L3 | `moved` / class-split | `unit/model_adaptation/` |
| `module/test_kimi_k25.py`, `module/test_kimi_k3.py` | L2 nightly MR | `moved` | `module/` |
| `model/test_deepseek_v32.py` | L1 | `moved`; nightly MLA compare is `module/test_deepseek_v32_performance.py` | `model/` |

## File-level inventory — CLI

| Current path | Layer | Action | Target |
|--------------|-------|--------|--------|
| `unit/test_comm_microbench_pure.py` | L3 | `keep` — SSOT for rank/topology/group | `unit/` |
| `module/test_generate_comm_microbench.py` | L2 orchestration | `moved`; duplicated pure classes stay for assertion compare | `module/` |
| `unit/test_op_replay_common_pure.py` | L3 | `keep` | `unit/` |
| `unit/test_op_replay_common.py` | L3 | `moved`; later merge into `*_pure.py` | `unit/` |
| `module/test_op_replay.py` | L2 CLI / adapter | `moved`; helper split later | `module/` |
| `module/test_start_microbench.py` | mix in L2 home | `moved`; parse vs NPU split later | `module/` |
| `module/test_throughput_optimizer.py` | L2 aggregation | `moved`; parse/draft-spec extracted | `module/` + `unit/test_throughput_optimizer_*.py` |
| `unit/test_fia_parser_backfill.py`, `unit/test_fia_common.py` | L3 | `keep` / `moved` from benchmark | `unit/` |
| `unit/test_cli_utils.py`, `unit/test_logo.py`, `unit/test_spec_cli.py`, `unit/test_grid_config.py`, `unit/test_shape_validation.py` | L3 | `keep` | `unit/` |
| `unit/test_main.py`, `unit/test_export.py`, `unit/test_compile.py` | L3 dispatch / flags | `moved` (reclassified from old L2 label) | `unit/` |
| `module/test_runner.py`, `module/test_image_generate.py`, `module/test_query_workloads.py` | L2 | `moved` | `module/` |
| `test_generate_shape_grid.py`, `test_query_driven_shape_grid.py`, `test_shape_grid_model_configs.py`, `test_replay_framework.py` | later | remaining flat mix | after this wave |

## File-level inventory — benchmark

| Current path | Layer | Action | Target |
|--------------|-------|--------|--------|
| `models/test_model_regression.py` + `models/cases/*.json` | L1 | `keep` — do not rebuild; do not edit baselines in this wave | stay |
| `ops/perf_database/test_op_mapping_schema.py` | operator-DB | `keep` | stay |
| `ops/perf_database/test_op_mapping_compile_passes.py` | operator-DB | `keep` | stay |
| `ops/perf_database/test_profiling_interpolation_non_regression.py` | operator-DB | `keep` | stay |
| `ops/perf_database/test_profiling_data_source.py` | mix | `keep` this wave — HCCL anchors stay with the operator-DB file | stay |
| `unit/test_mla_decomposition.py` | L3 | `moved` | TensorCast `unit/` |
| `unit/test_data_source.py` | L3 | `moved` | TensorCast `unit/` |
| `unit/test_empirical_database.py` | L3 | `moved` (kept separate from `unit/test_empirical.py`) | TensorCast `unit/` |
| `unit/test_empirical_metrics.py` | L3 | `moved` | TensorCast `unit/` |
| `unit/test_query_demand_capture.py` | L3 | `moved` | TensorCast `unit/` |
| `unit/test_fia_enriched_lookup.py` | L3 | `moved`; fixtures remain under benchmark and are referenced by absolute tests path | TensorCast `unit/` |
| `cli/unit/test_fia_common.py` | L3 | `moved` | CLI `unit/` |
| `unit/test_memory_estimator.py` | L3 | `moved` | TensorCast `unit/` |
| `cli/unit/test_trace_to_csv.py` | L3 | `moved` | CLI `unit/` |
| `cli/unit/test_extract_tc_from_chrome_trace.py` | L3 | `moved` | CLI `unit/` |
| `cli/unit/test_per_shape_comparison.py` | L3 | `moved` | CLI `unit/` |
| `cli/unit/test_compute_m6.py` | L3 | `moved` | CLI `unit/` |

## File-level inventory — smoke (this wave)

| Current path | Action |
|--------------|--------|
| `test_tensor_cast.py`, `test_compile_*.py`, `test_model_runner_compile_smoke.py`, `test_inference_smoke.py`, `test_fusion_passes_smoke.py`, `test_deepseek_v4_smoke.py`, `test_config_resolver_smoke.py` | `keep` as cheap guards; do not merge into L2 |
| `test_throughput_optimizer_smoke.py` | `review` 20–63 s nodes |
| `test_conftest_hygiene.py` | `keep` |
| `test_optix_optimizer_smoke.py`, `test_serving_cast.py`, `test_vllm_pd_smoke.py` | out of L1/L2/L3 file moves; leave in place |

## Keep-layout (no `unit/` / `module/` this wave)

| Tree | Action |
|------|--------|
| `tests/regression/model_diagnostics/` | Counted, keep-layout: **480** nodes (`e2e` 104, `application` 52 ≈ L2-duty; `specification` 149 + `domain` 50 + `organization` 46 + `comparison` 35 + `sources` 29 + `rendering` 12 + `integration` 2 + root 1 ≈ L3-duty). Default markers: 406 selected / 74 deselected (all `e2e` nightly/network). Smoke: `tests/smoke/test_model_diagnostics.py` (1). Do not add `unit/` / `module/`; `share-fixture` for e2e profile/case matrix only. Not part of TensorCast/CLI L2 798. |
| `tests/regression/scripts/` | `share-fixture` for `baseline` / sample `test_map` |
| `tests/helpers/tests/` | `keep` |
| `tests/regression/optix/`, `serving_cast/`, `web_ui/` | out of scope |

## Deletion candidates

**None approved.** Coverage run produced 76 `over_covered_symbol` and 149 Jaccard pairs (8 cross-file). All stay in `review` until a later PR records: kept node id, branch evidence, assertion table, `test_map` sole-watcher check, and the commands that passed L1/L2/L3 + smoke.

## PR 1 acceptance

- [x] Rules written in `ut_refactor.md` and `tests/README.md`
- [x] This inventory exists and can be sliced into later PRs
- [x] `pytest --collect-only` on the current tree: **4,605 collected / 4,391 selected** (214 deselected), same as the 2026-08-29 analysis baseline
- [x] No test file moved or deleted

## PR 2 acceptance

- [x] `tests/regression/tensor_cast/{unit,module}/` and `tests/regression/cli/{unit,module}/` exist; no nested `conftest.py`
- [x] First-wave whole-file L3 cases moved (TensorCast 25 files, CLI 8 files); mix/`split` files left in place
- [x] `test_pattern_match.py` uses absolute imports of component `conftest` / `test_common`
- [x] `test_interpolating_data_source.py` fixture path walks to `tests/` (`parents[3]`)
- [x] `pytest --collect-only` on the same trees as PR1: **4,605 collected / 4,391 selected** (214 deselected); node ids now include `…/unit/…`

## Full-tree responsibility audit

- Full tree at the audit point: **7,559 collected / 7,345 selected**; TensorCast: **1,851**, CLI: **908**, benchmark: **551**, smoke: **107**.
- Target L1: roughly 35–45 offline model-contract nodes, sourced from registered ModelProfiles plus supported generic and Diffusers families.
- Target L2: roughly 350–450 flagship depth nodes, including 120 nightly TensorCast nodes and the 22 JSON performance cases.
- L3 remains the majority (1,300+ TensorCast nodes); roughly 400 mixed-file nodes require class/node-level splits.

## L1 / L2 / L3 placement wave (this PR)

TensorCast flat directory now keeps only `conftest.py`, `test_common.py`, and `__init__.py`.

| Area | Done |
|------|------|
| L1 `model/` | Registry + text/VL/Diffusers contracts; lightweight `test_model_load.py` / `test_deepseek_v32.py` |
| L2 `module/` | Flagship MR, compile, TP/EP/quant/MTP/VL/video, nightly matrices |
| L3 `unit/` + `unit/model_adaptation/` | Pass/FX, helpers, model patches, DCP validation, image fixture contracts |
| CLI `module/` | Real optimizer / image / query / replay / microbench runs |
| CLI `unit/` | Parse, help, schema, draft-spec, and benchmark-ops L3 tools |
| benchmark `ops/` | Schema / compile-pass / interpolation / HCCL-mix stay; generic L3 moved |

Remaining later splits (do not block this wave):

- `tests/benchmark/ops/perf_database/test_profiling_data_source.py` HCCL anchors vs generic lookup
- CLI `test_generate_shape_grid.py`, `test_query_driven_shape_grid.py`, `test_shape_grid_model_configs.py`, `test_replay_framework.py`
- `module/test_generate_comm_microbench.py` overlapping pure classes vs `unit/test_comm_microbench_pure.py` (review assertions before delete)
- External `test_map` node-id refresh after merge (nightly `run_test_map_sync.sh` / `build_test_map`)

## PR 3 acceptance

- [x] Historical PR3 baseline used L1 accuracy / L2 flagship depth / L3 unit; superseded by the 2026-09-02 routing override above
- [x] TensorCast `model/` / `module/` / `unit/` and CLI `module/` / `unit/` populated; no nested `conftest.py`
- [x] Class-level splits for Kimi, DeepSeek-V4, MTP, DFC, SP rewriter, runtime BoundAnalyzer, DCP compile, optimizer parse
- [x] `pytest --collect-only` on TensorCast + CLI + benchmark + smoke: **3,484 collected / 3,344 selected** (140 deselected)
- [x] Nightly must refresh the external `test_map` after merge — file moves change pytest node ids; local trees do not write OBS `test_map.json`

## Seven-artifact migration acceptance

- [x] Design and authoring docs state model identity = L1, mechanism = L2, pure contract = L3
- [x] CLI is split by tested artifact; parser/validation/default/exit-code cases route to L3
- [x] All 35 v2 retained three-way-overlap cases are explicitly routed to L1 in the workbook
- [ ] Seven canonical builders exist under `tests/helpers/` and duplicated local builders are removed
- [ ] Real-model and pure-unit cases have moved out of every L2 physical directory
- [ ] L2 static import/model-id guard is enabled only after the previous item passes
- [ ] Affected tests pass; `run_test_map_sync.sh` refreshes `test_map`
- [ ] Existing `detect_redundant_cases` is wired into sync/nightly reporting and Jaccard warnings reach Feishu

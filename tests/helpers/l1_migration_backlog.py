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

"""Every remaining direct model construction, and why it stays or waits.

``keep`` means the call is the executor, the layer-model helper, the serving
runner, or a test of ``build_model`` itself. ``later`` means the call name is
still a direct constructor. The construction census test fails if a new direct
call is missing from this list. Node ids are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SiteNote:
    path: str
    function: str
    callee: str
    calls: int
    disposition: str
    reason: str


DIRECT_SITES: tuple[SiteNote, ...] = (

    SiteNote(
        "tests/benchmark/models/auto_baseline.py",
        "_run_single",
        "ModelRunner",
        1,
        "later",
        "The test still names a direct constructor. It was left because it tests that constructor or a patched class.",
    ),
    SiteNote(
        "tests/helpers/l1_scenario.py",
        "runner",
        "ModelRunner",
        1,
        "keep",
        "The executor constructs ModelRunner here. This call is the shared implementation, not a case to migrate.",
    ),
    SiteNote(
        "tests/helpers/model_builder.py",
        "build_or_get_cached_model",
        "build_model",
        1,
        "keep",
        "build_or_get_cached_model still calls build_model so its unit test can patch that name. Session caches use the executor.",
    ),
    SiteNote(
        "tests/helpers/model_builder.py",
        "layer_transformer",
        "TransformerModel",
        1,
        "keep",
        "layer_transformer builds from a prepared ModelConfig. The L1 executor only accepts UserInputConfig.",
    ),
    SiteNote(
        "tests/helpers/serving_runner.py",
        "serving_model_runner",
        "ModelRunner",
        1,
        "keep",
        "serving_cast ModelRunner takes a parallel config, a device, and a data-parallel rank. It is not the L1 executor.",
    ),
    SiteNote(
        "tests/helpers/tests/test_l1_scenario.py",
        "graph_fingerprint",
        "build_model",
        1,
        "keep",
        "The executor test fingerprints a direct build_model so the comparison stays outside the executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_dfc_default.py",
        "test_build_model_applies_user_dfc_config_before_compile",
        "build_model",
        1,
        "keep",
        "This test checks that build_model applies the user DFC flag before compile.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_model_load.py",
        "_get_model",
        "get_cached_build_model",
        1,
        "later",
        "The call still uses the session cache. That cache now builds the module through the L1 executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_mtp.py",
        "_get_model",
        "get_cached_build_model",
        1,
        "later",
        "The call still uses the session cache. That cache now builds the module through the L1 executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_mtp_ep.py",
        "_get_model",
        "get_cached_build_model",
        1,
        "later",
        "The call still uses the session cache. That cache now builds the module through the L1 executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_parallel_moe.py",
        "test_deepseek_with_ep",
        "get_cached_build_model",
        1,
        "later",
        "The call still uses the session cache. That cache now builds the module through the L1 executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_parallel_moe.py",
        "test_deepseek_with_redundant_experts_and_external_shared_expert",
        "get_cached_build_model",
        1,
        "later",
        "The call still uses the session cache. That cache now builds the module through the L1 executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_pipeline_parallel_model_builder.py",
        "test_build_model_allows_text_config_with_none_vision_config",
        "build_model",
        1,
        "keep",
        "build_or_get_cached_model still calls build_model so its unit test can patch that name. Session caches use the executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_pipeline_parallel_model_builder.py",
        "test_build_model_allows_vl_profile_with_language_path_for_pipeline",
        "build_model",
        1,
        "keep",
        "build_or_get_cached_model still calls build_model so its unit test can patch that name. Session caches use the executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_pipeline_parallel_model_builder.py",
        "test_build_model_keeps_mtp_only_on_last_pipeline_stage",
        "build_model",
        1,
        "keep",
        "build_or_get_cached_model still calls build_model so its unit test can patch that name. Session caches use the executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_pipeline_parallel_model_builder.py",
        "test_build_model_keeps_single_model_path_when_pp_size_is_one",
        "build_model",
        1,
        "keep",
        "build_or_get_cached_model still calls build_model so its unit test can patch that name. Session caches use the executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_pipeline_parallel_model_builder.py",
        "test_build_model_passes_compile_mode_to_pipeline_stages",
        "build_model",
        1,
        "keep",
        "build_or_get_cached_model still calls build_model so its unit test can patch that name. Session caches use the executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_pipeline_parallel_model_builder.py",
        "test_build_model_rejects_unsupported_pipeline_vl_without_language_path",
        "build_model",
        1,
        "keep",
        "build_or_get_cached_model still calls build_model so its unit test can patch that name. Session caches use the executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_pipeline_parallel_model_builder.py",
        "test_build_model_returns_real_split_pipeline_model_for_pp_size",
        "build_model",
        1,
        "keep",
        "build_or_get_cached_model still calls build_model so its unit test can patch that name. Session caches use the executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_pipeline_parallel_model_builder.py",
        "test_build_stage_model_clears_dflash_dspark_on_non_last_stage",
        "build_model",
        1,
        "keep",
        "build_or_get_cached_model still calls build_model so its unit test can patch that name. Session caches use the executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_pipeline_parallel_model_builder.py",
        "test_build_stage_model_clears_dspark_on_non_last_stage",
        "build_model",
        1,
        "keep",
        "build_or_get_cached_model still calls build_model so its unit test can patch that name. Session caches use the executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_pipeline_parallel_model_builder.py",
        "test_pipeline_model_draft_hf_config_exposed_from_last_stage",
        "build_model",
        1,
        "keep",
        "build_or_get_cached_model still calls build_model so its unit test can patch that name. Session caches use the executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_pipeline_parallel_model_builder.py",
        "test_pipeline_model_num_hidden_layers_includes_draft_layers",
        "build_model",
        1,
        "keep",
        "build_or_get_cached_model still calls build_model so its unit test can patch that name. Session caches use the executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_pipeline_parallel_model_builder.py",
        "test_pp_spec_decode_candidate_to_profile_contract",
        "build_model",
        1,
        "keep",
        "build_or_get_cached_model still calls build_model so its unit test can patch that name. Session caches use the executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_profiling_interpolation_phase1.py",
        "test_model_runner_disable_profiling_interpolation_uses_base_datasource",
        "ModelRunner",
        1,
        "keep",
        "The test patches model_runner_module.ModelRunner and constructs that class to see the profiling datasource wrapper.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_profiling_interpolation_phase1.py",
        "test_model_runner_wraps_profiling_datasource_by_default",
        "ModelRunner",
        1,
        "keep",
        "The test patches model_runner_module.ModelRunner and constructs that class to see the profiling datasource wrapper.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_quant_attention.py",
        "test_mla",
        "get_cached_build_model",
        1,
        "later",
        "The call still uses the session cache. That cache now builds the module through the L1 executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_quant_linear.py",
        "_get_model",
        "get_cached_build_model",
        1,
        "later",
        "The call still uses the session cache. That cache now builds the module through the L1 executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_repetition.py",
        "_run_test_deepseek_with_kvcache",
        "get_cached_build_model",
        1,
        "later",
        "The call still uses the session cache. That cache now builds the module through the L1 executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_runtime.py",
        "_get_model",
        "get_cached_build_model",
        1,
        "later",
        "The call still uses the session cache. That cache now builds the module through the L1 executor.",
    ),
    SiteNote(
        "tests/regression/tensor_cast/model/test_swiglu_fusion_pass.py",
        "_get_model",
        "get_cached_build_model",
        1,
        "later",
        "The call still uses the session cache. That cache now builds the module through the L1 executor.",
    ),
)


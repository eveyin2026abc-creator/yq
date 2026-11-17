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

"""One-layer L1 scenarios for the vendored model families."""

from __future__ import annotations

from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig

from tests.helpers.l1_scenario import L1RunArtifact, L1ScenarioExecutor

# Local fixtures already used by the diagnostics e2e cases. One entry per family.
FAMILY_MODEL_IDS = {
    "qwen3_dense": "tests/assets/model_config/qwen3_dense_0_6b",
    "qwen3_moe": "tests/assets/model_config/qwen3_moe_30b_a3b",
    "qwen3_vl": "tests/assets/model_config/qwen3_vl_8b_instruct",
    "qwen3_vl_moe": "tests/assets/model_config/qwen3_vl_moe_235b_a22b",
    "qwen3_5": "tests/assets/model_config/qwen3_5_27b",
    "qwen3_5_moe": "tests/assets/model_config/qwen3_5_moe_397b_a17b",
    "qwen3_next": "tests/assets/model_config/qwen3_next_80b_a3b",
    "deepseek_v3": "tests/assets/model_config/deepseek_v3",
    "deepseek_v32": "tests/assets/model_config/deepseek_v32",
    "deepseek_v4": "tests/assets/model_config/deepseek_v4_flash",
    "glm5": "tests/assets/model_config/glm5_1",
    "glm4": "tests/assets/model_config/glm4_7",
    "glm4v": "tests/assets/model_config/glm4v_9b_thinking",
    "glm4v_moe": "tests/assets/model_config/glm4v_moe_4_5v",
    "minimax_m2": "tests/assets/model_config/minimax_m2",
    "mimo_v2": "tests/assets/model_config/mimo_v2_flash",
    "ling": "tests/assets/model_config/ling_flash_2_0",
    "kimi_k25": "tests/assets/model_config/kimi_k2_5",
}


def family_input(model_id: str, **overrides) -> UserInputConfig:
    """Build a one-layer prefill for a vendored family config."""
    config = {
        "device": "TEST_DEVICE",
        "model_id": model_id,
        "num_queries": 2,
        "query_len": 10,
        "context_length": 0,
        "do_compile": False,
        "allow_graph_break": False,
        "quantize_linear_action": QuantizeLinearAction.DISABLED,
        "num_hidden_layers_override": 1,
    }
    config.update(overrides)
    return UserInputConfig(**config)


def run_family(executor: L1ScenarioExecutor, user_input: UserInputConfig) -> L1RunArtifact:
    """Run one family scenario on an existing executor."""
    return executor.run(user_input, generate_inputs_func=generate_inputs)

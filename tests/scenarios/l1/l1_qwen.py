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

"""Qwen scenarios that share one L1 build across assertion groups."""

from __future__ import annotations

from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig

from tests.helpers.l1_scenario import L1RunArtifact, L1ScenarioExecutor

QWEN3_32B = "Qwen/Qwen3-32B"


def qwen3_32b_input(**overrides) -> UserInputConfig:
    """Default Qwen3-32B prefill used by the shared L1 scenarios.

    ``num_hidden_layers_override`` defaults to 1 so the reuse check builds a
    tiny graph. Callers that need the full stack pass ``None``.
    """
    config = {
        "device": "TEST_DEVICE",
        "model_id": QWEN3_32B,
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


def run_qwen(executor: L1ScenarioExecutor, user_input: UserInputConfig) -> L1RunArtifact:
    """Run one Qwen scenario on an existing executor."""
    return executor.run(user_input, generate_inputs_func=generate_inputs)

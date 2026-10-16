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

"""Map a diagnostics run profile onto the L1 executor's user input.

Diagnostics e2e still captures through ``capture_artifact_for_profile``. This
module is the input half of a later move: the same profile becomes one
``UserInputConfig`` whose build signature follows the L1 field split. It does
not capture an artifact and it does not construct ``ModelRunner``.
"""

from __future__ import annotations

from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.model_config import WordEmbeddingTPMode
from tools.model_diagnostics.domain import ExecutionPhase
from tools.model_diagnostics.specification import DiagnosticsRunProfile

from tests.helpers.l1_scenario import L1BuildSignature, L1RunSignature


def user_input_from_profile(profile: DiagnosticsRunProfile) -> UserInputConfig:
    """Translate one diagnostics profile into the executor's input.

    ``context_length`` is a build field, so a missing decode context stays 0
    rather than becoming a second sentinel. ``decode`` itself is a run field:
    prefill and decode share a build when every build field matches.
    Parallel degrees multiply into ``world_size`` because the user config
    checks vision parallelism against that product.
    """

    parallel = profile.parallel
    world_size = parallel.tensor_parallel_size * parallel.pipeline_parallel_size * parallel.data_parallel_size
    embedding = None if profile.word_embedding_tp is None else WordEmbeddingTPMode(profile.word_embedding_tp)
    return UserInputConfig(
        model_id=profile.model_name,
        device=profile.device,
        num_queries=profile.batch_size,
        query_len=profile.query_length,
        context_length=0 if profile.context_length is None else profile.context_length,
        decode=profile.phase is ExecutionPhase.DECODE,
        num_mtp_tokens=profile.num_mtp_tokens,
        num_hidden_layers_override=profile.num_hidden_layers_override,
        do_compile=profile.do_compile,
        quantize_linear_action=QuantizeLinearAction(profile.quantize_linear_action),
        world_size=world_size,
        tp_size=parallel.tensor_parallel_size,
        pp_size=parallel.pipeline_parallel_size,
        dp_size=parallel.data_parallel_size,
        ep_size=parallel.expert_parallel_size,
        moe_dp_size=parallel.moe_data_parallel_size,
        enable_mlapo=profile.enable_mlapo,
        enable_redundant_experts=profile.enable_redundant_experts,
        enable_external_shared_experts=profile.enable_external_shared_experts,
        image_batch_size=profile.image_batch_size,
        image_height=profile.image_height,
        image_width=profile.image_width,
        word_embedding_tp=embedding,
    )


def build_signature_from_profile(profile: DiagnosticsRunProfile) -> L1BuildSignature:
    return L1BuildSignature.from_user_config(user_input_from_profile(profile))


def run_signature_from_profile(profile: DiagnosticsRunProfile) -> L1RunSignature:
    return L1RunSignature.from_user_config(user_input_from_profile(profile))

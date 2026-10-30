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

"""Diagnostics profiles become L1 signatures without capturing a model."""

from dataclasses import replace

from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tools.model_diagnostics.domain import ExecutionPhase, ParallelContext
from tools.model_diagnostics.specification import DiagnosticsRunProfile

from tests.helpers.l1_diagnostics import (
    build_signature_from_profile,
    run_signature_from_profile,
    user_input_from_profile,
)


def _profile(**overrides: object) -> DiagnosticsRunProfile:
    profile = DiagnosticsRunProfile(
        schema_version="1",
        model_name="tests/assets/model_config/qwen3_dense_0_6b",
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=2,
        query_length=10,
        context_length=None,
        num_mtp_tokens=0,
        parallel=ParallelContext(),
        selected_stage_regions=(),
        num_hidden_layers_override=1,
        do_compile=False,
        device="TEST_DEVICE",
        quantize_linear_action="DISABLED",
        word_embedding_tp=None,
    )
    return replace(profile, **overrides)


def test_prefill_profile_maps_onto_the_executor_input():
    user_input = user_input_from_profile(_profile())
    assert user_input.model_id.endswith("qwen3_dense_0_6b")
    assert user_input.num_queries == 2
    assert user_input.query_len == 10
    assert user_input.context_length == 0
    assert user_input.decode is False
    assert user_input.quantize_linear_action is QuantizeLinearAction.DISABLED
    assert user_input.world_size == 1


def test_query_length_stays_on_one_build_and_quantization_does_not():
    base = _profile()
    longer = _profile(query_length=4)
    quantized = _profile(quantize_linear_action="W8A8_DYNAMIC")
    assert build_signature_from_profile(base) == build_signature_from_profile(longer)
    assert build_signature_from_profile(base) != build_signature_from_profile(quantized)
    assert run_signature_from_profile(base) != run_signature_from_profile(longer)


def test_decode_flag_is_a_run_difference_when_context_matches():
    prefill = _profile(phase=ExecutionPhase.PREFILL, context_length=128)
    decode = _profile(phase=ExecutionPhase.DECODE, context_length=128)
    assert build_signature_from_profile(prefill) == build_signature_from_profile(decode)
    assert user_input_from_profile(decode).decode is True


def test_parallel_degrees_multiply_into_world_size():
    profile = _profile(parallel=ParallelContext(tensor_parallel_size=2, data_parallel_size=2, expert_parallel_size=2))
    user_input = user_input_from_profile(profile)
    assert user_input.world_size == 4
    assert user_input.tp_size == 2
    assert user_input.dp_size == 2
    assert user_input.ep_size == 2


def test_word_embedding_mode_is_copied():
    user_input = user_input_from_profile(_profile(word_embedding_tp="col"))
    assert user_input.word_embedding_tp == "col"


def test_diagnostics_user_input_uses_the_capture_contract():
    from tests.helpers.l1_diagnostics import diagnostics_user_input

    user_input = diagnostics_user_input(_profile())
    assert user_input.performance_model == ["analytic"]
    assert user_input.moe_tp_size is None
    assert user_input.quantize_linear_action is QuantizeLinearAction.DISABLED

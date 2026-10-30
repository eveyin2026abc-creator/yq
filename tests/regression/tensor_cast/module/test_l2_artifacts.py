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

"""The seven L2 artifact builders stay free of a model identity."""

from pathlib import Path

import pytest
from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction

from tests.helpers.l2_artifacts import (
    L2_KINDS,
    diagnostics_artifact,
    fx_artifact,
    microbench_artifact,
    parallel_artifact,
    quant_artifact,
    replay_artifact,
    runtime_event_artifact,
)
from tests.helpers.layer_guard import find_l2_model_violations_in_file


def test_seven_kinds_are_the_l2_set():
    artifacts = (
        quant_artifact(linear_action=QuantizeLinearAction.W8A8_DYNAMIC),
        parallel_artifact(world_size=4, tensor_parallel=2, expert_parallel=2),
        fx_artifact(),
        runtime_event_artifact(op_name="tensor_cast.linear", duration_us=10),
        diagnostics_artifact(region_id="output", status="PASS"),
        replay_artifact(num_queries=2, query_len=8, decode=True),
        microbench_artifact(op_name="aten.mm", shape=(2, 4), dtype="bfloat16"),
    )
    assert tuple(artifact.kind for artifact in artifacts) == L2_KINDS


def test_builders_reject_a_model_identity():
    with pytest.raises(TypeError, match="model identity"):
        quant_artifact(linear_action=QuantizeLinearAction.DISABLED, **{"model_id": "Qwen/Qwen3-32B"})


def test_parallel_artifact_rejects_an_uneven_world():
    with pytest.raises(ValueError, match="divisible"):
        parallel_artifact(world_size=4, tensor_parallel=3)


def test_fx_artifact_is_a_placeholder_graph():
    artifact = fx_artifact()
    assert artifact.node_ops == ("placeholder", "output")


def test_diagnostics_and_replay_reject_empty_contracts():
    with pytest.raises(ValueError, match="PASS or FAIL"):
        diagnostics_artifact(region_id="input", status="WARN")
    with pytest.raises(ValueError, match="positive"):
        replay_artifact(num_queries=0, query_len=4)


def test_l2_module_has_no_model_construction():
    import tests.helpers.l2_artifacts as l2_artifacts

    assert find_l2_model_violations_in_file(Path(l2_artifacts.__file__)) == ()

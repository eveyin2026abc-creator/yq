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

import inspect
from pathlib import Path

import pytest
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction

from tests.helpers.l2_artifacts import (
    L2_BUILDERS,
    L2_CONTRACTS,
    L2_KINDS,
    diagnostics_artifact,
    fx_artifact,
    microbench_artifact,
    parallel_artifact,
    quant_artifact,
    replay_artifact,
    runtime_event_artifact,
)
from tests.architecture.layer_guard import find_l2_model_violations_in_file


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


def test_contract_registry_is_complete_and_matches_python_signatures():
    assert tuple(contract.kind for contract in L2_CONTRACTS) == L2_KINDS
    assert tuple(L2_BUILDERS) == L2_KINDS
    for contract in L2_CONTRACTS:
        signature = inspect.signature(contract.builder)
        public = tuple(
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind is not inspect.Parameter.VAR_KEYWORD
        )
        assert public == contract.required_parameters + contract.optional_parameters
        for name in contract.required_parameters:
            assert signature.parameters[name].default is inspect.Parameter.empty
        for name in contract.optional_parameters:
            assert signature.parameters[name].default is not inspect.Parameter.empty


@pytest.mark.parametrize("contract", L2_CONTRACTS, ids=lambda contract: contract.kind)
def test_every_builder_rejects_model_identity(contract):
    required_values = {
        "linear_action": QuantizeLinearAction.DISABLED,
        "world_size": 1,
        "tensor_parallel": 1,
        "op_name": "aten.mm",
        "duration_us": 1,
        "region_id": "language",
        "status": "PASS",
        "num_queries": 1,
        "query_len": 1,
        "shape": (1,),
        "dtype": "fp16",
    }
    kwargs = {name: required_values[name] for name in contract.required_parameters}
    with pytest.raises(TypeError, match="model identity"):
        contract.builder(**kwargs, model_id="forbidden/model")


def test_builders_reject_a_model_identity():
    with pytest.raises(TypeError, match="model identity"):
        quant_artifact(linear_action=QuantizeLinearAction.DISABLED, **{"model_id": "Qwen/Qwen3-32B"})


def test_parallel_artifact_rejects_an_uneven_world():
    with pytest.raises(ValueError, match="must equal world_size"):
        parallel_artifact(world_size=4, tensor_parallel=3)


def test_quant_artifact_provides_product_config_and_minimal_linear_inputs():
    artifact = quant_artifact(
        linear_action=QuantizeLinearAction.W8A8_DYNAMIC,
        input_shape=(2, 4),
        output_features=3,
    )

    assert artifact.linear_config.quant_type.name == "W8A8"
    assert artifact.input_tensor.shape == (2, 4)
    assert artifact.weight.shape == (3, 4)


def test_parallel_artifact_provides_product_parallel_config():
    artifact = parallel_artifact(world_size=8, tensor_parallel=4, expert_parallel=2)

    assert artifact.config.world_size == 8
    assert artifact.config.tensor_parallel_size == 4
    assert artifact.config.expert_parallel_size == 2


def test_fx_artifact_is_a_placeholder_graph():
    artifact = fx_artifact()
    assert artifact.node_ops == ("placeholder", "output")
    assert artifact.graph_module("value") == "value"


def test_runtime_event_artifact_matches_runtime_capture_protocol():
    output = object()
    artifact = runtime_event_artifact(
        op_name="aten.mm.default",
        duration_us=10,
        args=("input",),
        kwargs={"weight": "weight"},
        out=output,
    )

    assert str(artifact.op_invoke_info.func) == "aten.mm.default"
    assert artifact.op_invoke_info.args == ("input",)
    assert artifact.op_invoke_info.kwargs == {"weight": "weight"}
    assert artifact.op_invoke_info.out is output


def test_diagnostics_and_replay_reject_empty_contracts():
    with pytest.raises(ValueError, match="invalid diagnostics status"):
        diagnostics_artifact(region_id="input", status="WARN")
    with pytest.raises(ValueError, match="positive"):
        replay_artifact(num_queries=0, query_len=4)


def test_replay_artifact_exposes_the_row_consumed_by_replay_components():
    artifact = replay_artifact(
        num_queries=2,
        query_len=16,
        fields={"Input Shapes": "2,16", "Average Duration(us)": "1.0"},
    )

    assert artifact.row == {
        "num_queries": 2,
        "query_len": 16,
        "decode": False,
        "Input Shapes": "2,16",
        "Average Duration(us)": "1.0",
    }
    assert artifact.trace[0] == {
        "event": "request",
        "num_queries": 2,
        "query_len": 16,
        "phase": "prefill",
    }
    assert artifact.trace[1] == {"event": "operator", "row": artifact.row}


def test_microbench_artifact_exposes_a_replay_row_request():
    artifact = microbench_artifact(op_name="ArgMaxV2", shape=(2, 4), dtype="DT_BF16")

    assert artifact.request == {
        "schema_version": 1,
        "kernel_type": "ArgMaxV2",
        "replay_row": {
            "Input Shapes": "2,4",
            "Input Data Types": "DT_BF16",
            "Input Formats": "ND",
            "Output Shapes": "2,4",
            "Output Data Types": "DT_BF16",
            "Output Formats": "ND",
        },
    }


def test_l2_module_has_no_model_construction():
    import tests.helpers.l2_artifacts as l2_artifacts

    assert find_l2_model_violations_in_file(Path(l2_artifacts.__file__)) == ()

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

"""Pure request, adapter, and output-shape rules for operator microbench."""

import pytest
from tools.perf_data_collection import run_op_microbench
from tools.perf_data_collection.op_microbench import adapters


def _request():
    return {
        "schema_version": 1,
        "kernel_type": "MatMulV2",
        "replay_row": {
            "Input Shapes": "1,4;2,4",
            "Input Data Types": "DT_BF16;DT_BF16",
            "Input Formats": "ND;ND",
            "Output Shapes": "1,2",
            "Output Data Types": "DT_BF16",
            "Output Formats": "ND",
        },
    }


def test_unregistered_operator_is_rejected_during_request_validation():
    request = _request()
    request["kernel_type"] = "ArbitraryPythonKernel"
    with pytest.raises(run_op_microbench.MicrobenchError) as error:
        run_op_microbench.normalize_request(request)
    assert error.value.code == "INVALID_REQUEST"
    assert "supported_operators" in error.value.details


def test_standalone_profiler_targets_include_master_alias_metadata():
    expected_aliases = {
        "BatchMatMulV2": "BatchMatMulNd",
        "ScatterNdUpdate": "ScatterNdUpdateAiCore",
        "Slice": "SliceAiCore",
        "Transpose": "TransposeAiCore",
    }
    for kernel_type, alias in expected_aliases.items():
        assert alias in adapters.ADAPTERS[kernel_type].target_op_types


def test_replay_spec_builds_default_and_overridden_contracts():
    default = adapters._replay_spec("ExampleOp", "torch.example")
    assert default == adapters.AdapterSpec(
        name="ExampleOp",
        api_path="torch.example",
        target_op_types=("ExampleOp",),
        replay_module="tools.perf_data_collection.op_replay.ExampleOp_run",
    )
    overridden = adapters._replay_spec(
        "ExampleOp", "torch.example", target_op_types=("PhysicalExampleOp",), is_composite=True
    )
    assert overridden.target_op_types == ("PhysicalExampleOp", "ExampleOp")
    assert overridden.is_composite is True


def test_scalar_and_zero_element_output_shapes_are_not_compatible():
    assert not run_op_microbench._shapes_compatible([], [0])
    assert not run_op_microbench._shapes_compatible([0], [])
    replay_row = {"Output Shapes": "[]", "Output Data Types": "DT_FLOAT", "Output Formats": "ND"}
    with pytest.raises(run_op_microbench.MicrobenchError) as error:
        run_op_microbench._validate_replay_outputs(
            replay_row,
            [{"logical_shape": [0], "dtype": "DT_FLOAT", "format": "ND"}],
        )
    assert error.value.code == "REPLAY_OUTPUT_MISMATCH"

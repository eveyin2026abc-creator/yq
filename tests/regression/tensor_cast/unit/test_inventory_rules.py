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

"""Layer rules stay in the order the register documents."""

from tests.helpers.inventory_rules import NodeRecord, classify


def test_rules_apply_in_order_and_do_not_label_the_rest_l2():
    smoke = classify(NodeRecord("tests/smoke/test_inference_smoke.py", ("nightly",), has_build_model=True))
    assert smoke.target_layer == "smoke"
    assert smoke.migration == "保持"
    assert smoke.schedule == "nightly"

    benchmark_model = classify(NodeRecord("tests/benchmark/models/test_model_regression.py", ("nightly",)))
    assert benchmark_model.target_layer == "L1"
    assert benchmark_model.migration == "保持"
    assert benchmark_model.schedule == "benchmark"
    assert benchmark_model.assertion_overlap == "numeric"

    benchmark_op = classify(NodeRecord("tests/benchmark/ops/test_mm.py", ("npu",)))
    assert benchmark_op.target_layer == "benchmark-op"
    assert benchmark_op.schedule == "benchmark"

    e2e = classify(NodeRecord("tests/regression/model_diagnostics/e2e/test_qwen.py", ("network",)))
    assert e2e.target_layer == "L1"
    assert e2e.migration == "待归并到 L1"
    assert e2e.assertion_overlap == "diagnostics"
    assert e2e.schedule == "nightly-network"

    helper = classify(NodeRecord("tests/helpers/tests/test_l3_contracts.py", has_model_runner=True))
    assert helper.target_layer == "L3"
    assert helper.migration == "保持"

    runner = classify(NodeRecord("tests/regression/tensor_cast/test_text_generate.py", has_model_runner=True))
    assert runner.target_layer == "L1"
    assert runner.migration == "待归并到 L1"
    assert runner.assertion_overlap == "未判定"
    assert runner.build_signature

    both = classify(
        NodeRecord("tests/regression/tensor_cast/test_runtime.py", ("npu",), has_model_runner=True, has_build_model=True)
    )
    assert both.evidence == "file text contains ModelRunner and build_model"
    assert both.schedule == "excluded-npu"

    unknown = classify(NodeRecord("tests/regression/tensor_cast/unit/test_layer_guard.py"))
    assert unknown.target_layer == "待人工确认"
    assert unknown.migration == "待人工确认"
    assert unknown.build_signature == ""

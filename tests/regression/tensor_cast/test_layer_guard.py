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

"""The L2 scanner reports model construction and stays silent on a clean module."""

from tests.helpers.layer_guard import find_l2_model_violations


def test_scanner_reports_model_construction():
    source = "def build(model_id):\n    return ModelRunner(model_id=model_id)\n"
    violations = find_l2_model_violations(source)
    assert any("parameter model_id" in item for item in violations)
    assert any("uses ModelRunner" in item for item in violations)
    assert any("keyword model_id" in item for item in violations)


def test_scanner_ignores_a_mention_in_a_string():
    source = 'MESSAGE = "do not call ModelRunner"\n\ndef build():\n    return MESSAGE\n'
    assert find_l2_model_violations(source) == ()


def test_scanner_reports_build_model_and_transformer_model():
    source = "def build():\n    build_model(config)\n    TransformerModel(model_id='x')\n"
    violations = find_l2_model_violations(source)
    assert any("build_model" in item for item in violations)
    assert any("TransformerModel" in item for item in violations)

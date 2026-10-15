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

"""Vendored model families share one L1 build per workload."""

import pytest
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction

from tests.helpers.l1_families import FAMILY_MODEL_IDS, family_input, run_family
from tests.helpers.l1_scenario import L1ScenarioExecutor


@pytest.fixture(params=tuple(FAMILY_MODEL_IDS.items()), ids=tuple(FAMILY_MODEL_IDS))
def family_model(request):
    name, model_id = request.param
    return name, model_id


def test_same_scenario_builds_once(family_model):
    _name, model_id = family_model
    executor = L1ScenarioExecutor()
    try:
        user_input = family_input(model_id)
        first = run_family(executor, user_input)
        second = run_family(executor, user_input)
        assert first is second
        assert executor.counters.build_miss == 1
        assert executor.counters.forward_count == 1
        assert executor.counters.run_hit == 1
        executor.runner(user_input)
        assert executor.counters.build_hit == 1
        assert first.operator_names()
    finally:
        executor.reset()


def test_query_length_reuses_the_build(family_model):
    _name, model_id = family_model
    executor = L1ScenarioExecutor()
    try:
        run_family(executor, family_input(model_id, query_len=10))
        run_family(executor, family_input(model_id, query_len=4))
        assert executor.counters.build_miss == 1
        assert executor.counters.forward_count == 2
    finally:
        executor.reset()


def test_quantization_is_a_separate_build(family_model):
    _name, model_id = family_model
    executor = L1ScenarioExecutor()
    try:
        run_family(executor, family_input(model_id))
        run_family(
            executor,
            family_input(model_id, quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC),
        )
        assert executor.counters.build_miss == 2
        assert executor.counters.forward_count == 2
    finally:
        executor.reset()

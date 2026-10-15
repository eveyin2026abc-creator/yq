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

"""Qwen3-32B scenarios share one build and one forward per workload."""

import unittest

from tensor_cast.core.quantization.datatypes import QuantizeLinearAction

from tests.helpers.l1_qwen import qwen3_32b_input, run_qwen
from tests.helpers.l1_scenario import L1ScenarioExecutor


class TestQwenL1Reuse(unittest.TestCase):
    def setUp(self):
        self.executor = L1ScenarioExecutor()

    def tearDown(self):
        self.executor.reset()

    def test_same_scenario_builds_once_and_runs_once(self):
        user_input = qwen3_32b_input()
        first = run_qwen(self.executor, user_input)
        second = run_qwen(self.executor, user_input)

        self.assertIs(first, second)
        self.assertEqual(self.executor.counters.build_miss, 1)
        self.assertEqual(self.executor.counters.forward_count, 1)
        self.assertEqual(self.executor.counters.run_hit, 1)
        self.executor.runner(user_input)
        self.assertEqual(self.executor.counters.build_hit, 1)
        exec_time = first.metrics.execution_time_s
        if isinstance(exec_time, dict):
            exec_time = next(iter(exec_time.values()))
        self.assertGreater(exec_time, 0)
        self.assertTrue(first.operator_names())

    def test_query_length_reuses_the_build(self):
        run_qwen(self.executor, qwen3_32b_input(query_len=10))
        run_qwen(self.executor, qwen3_32b_input(query_len=4))

        self.assertEqual(self.executor.counters.build_miss, 1)
        self.assertEqual(self.executor.counters.forward_count, 2)

    def test_quantization_is_a_separate_build(self):
        run_qwen(self.executor, qwen3_32b_input())
        run_qwen(
            self.executor,
            qwen3_32b_input(quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC),
        )

        self.assertEqual(self.executor.counters.build_miss, 2)
        self.assertEqual(self.executor.counters.forward_count, 2)

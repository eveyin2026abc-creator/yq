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

"""Construct serving_cast's ModelRunner.

L1_EXECUTION_EXEMPT: serving_cast's runner is the subject under test and cannot
be executed by tensor_cast's L1ScenarioExecutor.

This is not ``tensor_cast.core.model_runner.ModelRunner`` and it does not go
through ``L1ScenarioExecutor``. The constructor takes a serving parallel config,
a device, and a data-parallel rank.
"""

from __future__ import annotations

from serving_cast.model_runner import ModelRunner


def serving_model_runner(parallel_config, device: str = "TEST_DEVICE", *, dp_rank: int = 0) -> ModelRunner:
    return ModelRunner(parallel_config, device, dp_rank=dp_rank)

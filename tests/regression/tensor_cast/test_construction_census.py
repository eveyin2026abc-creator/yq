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

"""The construction census sees real calls and ignores the guard's string fixtures."""

from tests.helpers.construction_census import iter_sites, render_census, sites_in


def test_executor_and_direct_sites_are_both_visible():
    sites = iter_sites()
    smoke = sites_in("tests/smoke/test_inference_smoke.py", sites)
    assert any(site.callee == "isolated_l1_executor" for site in smoke)
    runtime = sites_in("tests/regression/tensor_cast/test_runtime.py", sites)
    assert not any(site.callee == "build_model" for site in runtime)
    pipeline = sites_in("tests/regression/tensor_cast/test_pipeline_parallel_model_builder.py", sites)
    assert any(site.callee == "build_model" for site in pipeline)
    e2e = sites_in("tests/regression/model_diagnostics/e2e/conftest.py", sites)
    assert not any(site.callee == "capture_artifact_for_profile" for site in e2e)
    integration = sites_in("tests/regression/model_diagnostics/sources/test_runtime_capture_integration.py", sites)
    assert any(site.callee == "capture_artifact_for_profile" for site in integration)


def test_l2_builders_and_the_guard_fixture_are_not_model_calls():
    sites = iter_sites()
    l2 = sites_in("tests/helpers/l2_artifacts.py", sites)
    assert l2 == ()
    guard = sites_in("tests/regression/tensor_cast/test_layer_guard.py", sites)
    assert guard == ()


def test_census_text_counts_the_same_sites():
    sites = iter_sites()
    text = render_census(sites)
    assert f"Direct model constructions: {sum(site.direct_model for site in sites)}." in text
    assert f"Executor constructions: {sum(site.executor for site in sites)}." in text
    assert "| tests/helpers/l1_scenario.py |" in text

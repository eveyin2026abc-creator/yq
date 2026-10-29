"""Offline, real-build acceptance scenarios for the four L1 migration batches."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import tensor_cast.ops  # noqa: F401 -- register custom operators before model build
from tensor_cast.core.user_config import UserInputConfig
from tests.helpers.l1_scenario import L1ScenarioExecutor
from tests.helpers.model_assets import vendored_model_config_path

# L1_FAMILY: qwen
# L1_FAMILY: dense
# L1_FAMILY: moe
# L1_FAMILY: multimodal


@dataclass(frozen=True)
class FamilyScenario:
    name: str
    family: str
    model_id: str
    model_type: str
    capability_operator: str
    vision: bool = False


SCENARIOS = (
    FamilyScenario("qwen25-7b", "qwen", "Qwen/Qwen2.5-7B", "qwen2", "tensor_cast.attention.default"),
    FamilyScenario("dense-qwen3-06b", "dense", "Qwen/Qwen3-0.6B", "qwen3", "tensor_cast.attention.default"),
    FamilyScenario("moe-qwen3-30b", "moe", "Qwen/Qwen3-30B-A3B", "qwen3_moe", "tensor_cast.init_routing_v2.default"),
    FamilyScenario(
        "multimodal-qwen3vl-8b",
        "multimodal",
        "Qwen/Qwen3-VL-8B-Instruct",
        "qwen3_vl",
        "tensor_cast.attention.default",
        vision=True,
    ),
    FamilyScenario("dense-qwen3-8b", "dense", "Qwen/Qwen3-8B", "qwen3", "tensor_cast.attention.default"),
    FamilyScenario("moe-deepseek-v3", "moe", "deepseek-ai/DeepSeek-V3", "deepseek_v3", "tensor_cast.multihead_latent_attention.default"),
    FamilyScenario("moe-deepseek-v32", "moe", "deepseek-ai/DeepSeek-V3.2", "deepseek_v32", "tensor_cast.mla_sparse_attention.default"),
    FamilyScenario("moe-kimi-k2", "moe", "moonshotai/Kimi-K2-Thinking", "deepseek_v3", "tensor_cast.multihead_latent_attention.default"),
    FamilyScenario("moe-minimax-m2", "moe", "MiniMaxAI/MiniMax-M2", "minimax_m2", "tensor_cast.init_routing_v2.default"),
    FamilyScenario("multimodal-glm45v", "multimodal", "zai-org/GLM-4.5V", "glm4v_moe", "tensor_cast.attention.default", True),
    FamilyScenario(
        "multimodal-qwen3vl-moe",
        "multimodal",
        "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "qwen3_vl_moe",
        "tensor_cast.init_routing_v2.default",
        True,
    ),
)

INCREMENTAL_SCENARIOS = frozenset(
    {"qwen25-7b", "dense-qwen3-06b", "moe-qwen3-30b", "multimodal-qwen3vl-8b"}
)

PARAMETERS = tuple(
    pytest.param(
        scenario,
        id=scenario.name,
        marks=() if scenario.name in INCREMENTAL_SCENARIOS else pytest.mark.nightly,
    )
    for scenario in SCENARIOS
)


def _execution_times(metrics) -> list[float]:
    value = metrics.execution_time_s
    if isinstance(value, dict):
        return [float(item) for item in value.values()]
    return [float(value)]


def _assert_scenario(scenario: FamilyScenario) -> None:
    """One real vendored model build/forward feeds all three assertion groups."""

    config = UserInputConfig(
        model_id=vendored_model_config_path(scenario.model_id),
        device="TEST_DEVICE",
        num_queries=1,
        query_len=8,
        context_length=8,
        num_hidden_layers_override=1,
    )
    executor = L1ScenarioExecutor()

    artifact = executor.run(config)
    cached = executor.run(config)
    operators = artifact.operator_names()

    # Numeric assertion group.
    assert _execution_times(artifact.metrics)
    assert all(value >= 0.0 for value in _execution_times(artifact.metrics))

    # Structural assertion group.
    assert cached is artifact
    assert artifact.model is not None
    assert operators
    assert executor.counters.build_miss == 1
    assert executor.counters.forward_count == 1
    assert executor.counters.run_hit == 1

    # Capability assertion group.
    assert artifact.model.model_config.hf_config.model_type == scenario.model_type
    assert scenario.capability_operator in operators
    assert bool(artifact.model.is_vl_model) is scenario.vision


def _family_parameters(family: str):
    return tuple(param for param, scenario in zip(PARAMETERS, SCENARIOS) if scenario.family == family)


@pytest.mark.parametrize("scenario", _family_parameters("qwen"))
def test_qwen_scenarios(scenario: FamilyScenario) -> None:
    _assert_scenario(scenario)


@pytest.mark.parametrize("scenario", _family_parameters("dense"))
def test_dense_scenarios(scenario: FamilyScenario) -> None:
    _assert_scenario(scenario)


@pytest.mark.parametrize("scenario", _family_parameters("moe"))
def test_moe_scenarios(scenario: FamilyScenario) -> None:
    _assert_scenario(scenario)


@pytest.mark.parametrize("scenario", _family_parameters("multimodal"))
def test_multimodal_scenarios(scenario: FamilyScenario) -> None:
    _assert_scenario(scenario)

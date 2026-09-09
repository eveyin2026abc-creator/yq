"""Pilot for the model-level (L1) execution model, on one Qwen scenario.

The same Qwen prefill path used to be built and run three times: once to assert
performance numbers, once to assert operator structure, and once to capture a
diagnostics artifact. Here it is built once and run once, and all three
assertion groups read that single result.

``test_three_assertion_groups_cost_one_build_and_one_forward`` is the gate this
pilot exists to prove. ``test_shared_forward_matches_a_dedicated_capture_forward``
pins the premise that makes it possible: the observer receives the ``Runtime``
after ``Runtime.__exit__``, so diagnostics evidence is already complete and does
not need a forward pass of its own.
"""

from __future__ import annotations

import collections

import pytest

import tensor_cast.ops  # noqa: F401 — register custom ops before building
from tensor_cast.core.input_generator import generate_inputs
from tensor_cast.core.user_config import UserInputConfig
from tests.helpers.l1_scenario import L1ScenarioExecutor
from tools.model_diagnostics.domain import ExecutionPhase
from tools.model_diagnostics.domain.artifact import ProducerInfo
from tools.model_diagnostics.domain.models import ModelRunContext, ParallelContext
from tools.model_diagnostics.sources.runtime_capture import (
    RuntimeArtifactCapture,
    capture_model_runner_artifact,
)

MODEL_ID = "Qwen/Qwen3-32B"
QUERY_LEN = 32
CONTEXT_LENGTH = 32
BLOCK_SIZE = 128
EXPECTED_NUM_HIDDEN_LAYERS = 64
# Qwen3-32B applies seven quantized linear projections per decoder layer.
QUANT_LINEARS_PER_LAYER = 7
EXPECTED_WEIGHT_SIZE_GB = 31.981241464614868
# 64 layers x 8 KV heads x 128 head dim x 2 tensors x 2 bytes = 256 KiB per token.
EXPECTED_KV_CACHE_PER_TOKEN_GB = 0.000244140625

CAPTURE_BACKEND = "tensor_cast.runtime_observer"


def make_user_config() -> UserInputConfig:
    return UserInputConfig(
        model_id=MODEL_ID,
        device="TEST_DEVICE",
        num_queries=1,
        query_len=QUERY_LEN,
        context_length=CONTEXT_LENGTH,
        block_size=BLOCK_SIZE,
    )


def make_run_context() -> ModelRunContext:
    return ModelRunContext(
        model_name=MODEL_ID,
        entrypoint="text_generate",
        phase=ExecutionPhase.PREFILL,
        batch_size=1,
        query_length=QUERY_LEN,
        context_length=CONTEXT_LENGTH,
        parallel=ParallelContext(),
        model_config={},
        quantization_config={},
    )


def make_producer() -> ProducerInfo:
    return ProducerInfo(package_version="local", git_revision=None, capture_backend=CAPTURE_BACKEND)


@pytest.fixture(scope="module")
def prefill_artifact(l1_executor):
    """The one Qwen prefill result that every assertion group below reads."""
    return l1_executor.run(make_user_config(), generate_inputs_func=generate_inputs)


# --- assertion group 1: performance numbers ---------------------------------


def test_weight_and_kv_cache_sizes(prefill_artifact):
    metrics = prefill_artifact.metrics
    assert metrics.model_weight_size_gb == pytest.approx(EXPECTED_WEIGHT_SIZE_GB, rel=1e-6)
    assert metrics.kv_cache_per_token_gb == pytest.approx(EXPECTED_KV_CACHE_PER_TOKEN_GB, rel=1e-9)
    # The cache is allocated in whole blocks, so a 32-token request still pays
    # for one full block.
    assert metrics.kv_cache_size_gb == pytest.approx(EXPECTED_KV_CACHE_PER_TOKEN_GB * BLOCK_SIZE, rel=1e-9)


def test_memory_accounting_is_self_consistent(prefill_artifact):
    metrics = prefill_artifact.metrics
    assert metrics.peak_memory_usage_gb == pytest.approx(
        metrics.model_weight_size_gb + metrics.kv_cache_size_gb + metrics.model_activation_size_gb,
        rel=1e-9,
    )
    assert metrics.device_memory_available_gb == pytest.approx(
        metrics.total_device_memory_gb - metrics.peak_memory_usage_gb - metrics.reserved_memory_gb,
        rel=1e-9,
    )


def test_throughput_follows_from_execution_time(prefill_artifact):
    metrics = prefill_artifact.metrics
    assert set(metrics.execution_time_s) == {"analytic"}
    execution_time_s = metrics.execution_time_s["analytic"]
    assert execution_time_s > 0
    assert metrics.tps_per_model["analytic"] == pytest.approx(QUERY_LEN / execution_time_s, rel=1e-9)


# --- assertion group 2: operator structure ----------------------------------


def test_operator_structure_covers_every_layer(prefill_artifact):
    counts = collections.Counter(prefill_artifact.operator_names())
    assert prefill_artifact.model.hf_config.num_hidden_layers == EXPECTED_NUM_HIDDEN_LAYERS
    assert counts["tensor_cast.static_quant_linear.default"] == QUANT_LINEARS_PER_LAYER * EXPECTED_NUM_HIDDEN_LAYERS
    # Quantized linears are fed by a dynamic quantize of their activation.
    assert counts["tensor_cast.dynamic_quantize_symmetric.default"] == counts["tensor_cast.static_quant_linear.default"]


def test_runtime_events_describe_the_same_run_as_the_metrics(prefill_artifact):
    assert prefill_artifact.operator_names()
    assert len(prefill_artifact.runtime_event_list) > 0
    assert prefill_artifact.table_result is not None


# --- assertion group 3: diagnostics capability ------------------------------


def test_diagnostics_artifact_comes_from_the_shared_forward(prefill_artifact):
    artifact = RuntimeArtifactCapture.snapshot(
        prefill_artifact.runtime,
        run_context=make_run_context(),
        producer=make_producer(),
    )
    assert len(artifact.operator_calls) == len(prefill_artifact.operator_names())
    assert artifact.producer.capture_backend == CAPTURE_BACKEND
    assert all(call.tensors for call in artifact.operator_calls)


# --- the gate ----------------------------------------------------------------


def test_three_assertion_groups_cost_one_build_and_one_forward():
    """Three assertion groups over one scenario must not run the model twice."""
    executor = L1ScenarioExecutor()
    user_config = make_user_config()

    numbers = executor.run(user_config, generate_inputs_func=generate_inputs)
    structure = executor.run(user_config, generate_inputs_func=generate_inputs)
    diagnostics = executor.run(user_config, generate_inputs_func=generate_inputs)

    assert executor.counters.build_miss == 1
    assert executor.counters.forward_count == 1
    assert executor.counters.run_hit == 2
    assert executor.counters.forwards_by_build[numbers.build_signature.describe()] == 1
    assert numbers is structure is diagnostics


def test_shared_forward_matches_a_dedicated_capture_forward(prefill_artifact, l1_executor):
    """Reusing the run_inference Runtime must lose no diagnostics evidence.

    If this ever fails, the executor's single-forward design is wrong and must be
    revised rather than papered over by letting diagnostics run its own forward.
    """
    shared = RuntimeArtifactCapture.snapshot(
        prefill_artifact.runtime,
        run_context=make_run_context(),
        producer=make_producer(),
    )
    dedicated = capture_model_runner_artifact(
        l1_executor.runner(make_user_config()),
        generate_inputs_func=generate_inputs,
        run_context=make_run_context(),
        producer=make_producer(),
    )

    assert [call.operator_name for call in shared.operator_calls] == [
        call.operator_name for call in dedicated.operator_calls
    ]
    assert [call.tensors for call in shared.operator_calls] == [call.tensors for call in dedicated.operator_calls]


# --- the reuse contract ------------------------------------------------------


def test_reused_runner_serves_the_current_workload_not_the_first_one():
    """A runner cached by build signature must be rebound to each workload.

    run_inference reads num_queries, query_len and block_size off
    runner.user_input, so without rebinding the second scenario would silently
    report the first scenario's batch.
    """
    executor = L1ScenarioExecutor()
    single = executor.run(make_user_config(), generate_inputs_func=generate_inputs)

    batched_config = make_user_config()
    batched_config.num_queries = 4
    batched = executor.run(batched_config, generate_inputs_func=generate_inputs)

    assert executor.counters.build_miss == 1, "changing a run field must not trigger a rebuild"
    assert executor.counters.forward_count == 2
    assert single.metrics.batch_size == 1
    assert batched.metrics.batch_size == 4
    assert batched.metrics.model_weight_size_gb == pytest.approx(single.metrics.model_weight_size_gb, rel=1e-9)

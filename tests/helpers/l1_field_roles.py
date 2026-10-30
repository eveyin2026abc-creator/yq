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

"""Reviewed role of every ``UserInputConfig`` field in the L1 build signature.

The role must match ``RUN_FIELDS`` and ``OBSERVABILITY_FIELDS`` in
``l1_scenario``. A field that is in neither set is a build field. The note says
why that split is safe for reuse. Adding a field to ``UserInputConfig`` without
a note fails the field-role test.
"""

from __future__ import annotations

from tests.helpers.l1_scenario import OBSERVABILITY_FIELDS, RUN_FIELDS

FIELD_NOTES: dict[str, str] = {
    "device": "The device profile selects kernels and bandwidths, so it is part of the built graph.",
    "model_id": "The checkpoint identity selects the architecture and weights layout.",
    "num_queries": "Batch size changes the request, not the built module.",
    "query_len": "Sequence length is a run input. The same compiled graph serves more than one length.",
    "context_length": "Context length changes KV allocation and is rebuilt rather than rebound.",
    "prefix_cache_hit_rate": "The hit rate scales a run. It does not change the module.",
    "enable_decode_kv_offload": "Offload inserts a different KV path into the built graph.",
    "decode_offload_buffer_multiplier": "The multiplier sizes the offload buffer inside the build.",
    "decode_offload_hit_rate": "The offload hit rate is a numeric input to that built path.",
    "host_memory_available_gb": "Host memory changes whether the offload path is legal for this build.",
    "enable_prefill_kv_offload": "Prefill offload is a different graph from a resident KV cache.",
    "do_compile": "Compile and eager are different built forms of the same config.",
    "allow_graph_break": "Graph-break policy changes what torch.compile keeps in the graph.",
    "dynamic_shapes": "Static and dynamic shapes are different compilation signatures.",
    "dump_input_shapes": "Shape dumps are an observability side channel.",
    "dump_op_bound_results": "Operator-bound dumps are an observability side channel.",
    "chrome_trace": "The trace path does not change the graph or the numeric metrics.",
    "graph_log_url": "The graph log destination does not change the graph.",
    "log_level": "Log verbosity does not change the graph.",
    "quantize_linear_action": "Linear quantization replaces modules, so it is a new build.",
    "quantize_non_expert_linear_action": "Non-expert quantization is a separate module replacement.",
    "quantize_lmhead": "Quantizing the LM head changes the built head.",
    "mxfp4_group_size": "The MXFP4 group size is part of the quant module layout.",
    "quantize_attention_action": "Attention quantization replaces attention modules.",
    "enable_multistream": "Multistream changes the compiled execution schedule.",
    "enable_sequence_parallel": "Sequence parallel is a compilation pass and must be passed explicitly.",
    "enable_matmul_allreduce": "Matmul-allreduce is a compilation pass and must be passed explicitly.",
    "enable_dispatch_ffn_combine": "Dispatch-FFN-combine is copied from the user input during build.",
    "enable_mlapo": "MLAPO changes the built attention path.",
    "decode": "Prefill versus decode is a run mode. Context length, when it differs, is the build field.",
    "num_mtp_tokens": "MTP width changes the built speculative head.",
    "mtp_acceptance_rate": "Acceptance rates scale a run. They do not rebuild the MTP head.",
    "speculative_method": "The speculative method selects MTP, DFlash, or DSPark at build time.",
    "num_speculative_tokens": "The speculative width is bridged into the built MTP graph.",
    "num_speculative_tokens_explicit": "An explicit speculative width changes the built MTP graph.",
    "acceptance_length": "Acceptance length is a property of the built speculative config.",
    "num_draft_layers": "Draft depth changes the built draft model.",
    "draft_model_config_path": "A different draft config is a different built model.",
    "dspark_markov_rank": "The DSPark rank is part of the built draft module.",
    "dspark_markov_head": "The DSPark head variant is part of the built draft module.",
    "num_hidden_layers_override": "The layer override truncates the built stack.",
    "disable_repetition": "Repetition changes how the built layers are shared.",
    "reserved_memory_gb": "Reserved memory is a run budget, not a graph change.",
    "world_size": "World size selects the parallel layout of the build.",
    "tp_size": "Tensor parallel splits the built projections.",
    "pp_size": "Pipeline parallel splits the built stage graph.",
    "dp_size": "Data parallel changes the built device mesh.",
    "pp_layer_partition": "An explicit pipeline partition is part of the built stage graph.",
    "microbatch_size": "Microbatch size is part of the pipeline schedule baked into the build.",
    "o_proj_tp_size": "A custom output-projection shard is a different build.",
    "o_proj_dp_size": "A custom output-projection data parallel degree is a different build.",
    "mlp_tp_size": "A custom MLP shard is a different build.",
    "mlp_dp_size": "A custom MLP data parallel degree is a different build.",
    "lmhead_tp_size": "A custom LM-head shard is a different build.",
    "lmhead_dp_size": "A custom LM-head data parallel degree is a different build.",
    "ep_size": "Expert parallel changes the built MoE layout.",
    "moe_dp_size": "MoE data parallel changes the built expert mesh.",
    "moe_tp_size": "MoE tensor parallel changes the built expert shards.",
    "dcp_size": "Context parallel changes the built attention layout.",
    "word_embedding_tp": "The embedding parallel mode changes the built embedding.",
    "enable_redundant_experts": "Redundant experts change the built MoE modules.",
    "enable_shared_expert_tp": "Shared-expert tensor parallel changes the built MoE modules.",
    "enable_external_shared_experts": "External shared experts change the built MoE modules.",
    "host_external_shared_experts": "Host placement of shared experts changes the built MoE modules.",
    "vision_tp_size": "Vision tensor parallel changes the built visual tower.",
    "block_size": "Block size changes the paged-cache run, not the built module.",
    "remote_source": "The weight source selects how the build loads the checkpoint.",
    "image_batch_size": "Image batch is a run input to an already built visual tower.",
    "image_height": "Image height is a run input. The visual tower is already built.",
    "image_width": "Image width is a run input. The visual tower is already built.",
    "performance_model": "The performance model is rebound on a reused runner.",
    "profiling_database": "The profiling database is rebound on a reused runner.",
    "disable_profiling_interpolation": "Interpolation is a performance-model switch, not a new graph.",
    "analytic_calibration_profile": "The calibration profile changes the analytic model attached at build.",
    "analytic_calibration_stack": "The calibration stack selects which profile the build loads.",
}


def field_role(name: str) -> str:
    """Return ``build``, ``run``, or ``observability`` for one config field."""

    if name in RUN_FIELDS:
        return "run"
    if name in OBSERVABILITY_FIELDS:
        return "observability"
    if name in FIELD_NOTES:
        return "build"
    raise KeyError(name)

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

"""Shared execution skeleton for MTP and MTP+EP L1 scenarios."""

import torch
from tensor_cast.layers.sampler import SamplingMetadata
from tensor_cast.patch_torch import patch_torch

_UNSET = object()


def run_mtp_forward(
    model,
    *,
    num_tokens: int,
    batch_size: int,
    attention_meta=None,
    kv_cache_by_layers=None,
    selected_token_indices=_UNSET,
):
    """Run one meta-device MTP forward while preserving each caller's metadata contract."""
    inputs = torch.empty([batch_size, num_tokens], dtype=torch.long, device="meta")
    position_ids = torch.empty([batch_size, num_tokens], dtype=torch.long, device="meta")
    sampling_kwargs = {}
    if attention_meta is not None:
        sampling_kwargs["query_start_loc"] = attention_meta.query_start_loc
    if selected_token_indices is not _UNSET:
        sampling_kwargs["selected_token_indices"] = selected_token_indices
    forward_kwargs = {"sampling_metadata": SamplingMetadata(**sampling_kwargs)}
    if attention_meta is not None:
        forward_kwargs["attention_meta"] = attention_meta
        forward_kwargs["kv_cache_by_layers"] = kv_cache_by_layers
    with torch.no_grad(), patch_torch():
        return model.forward(inputs, position_ids, **forward_kwargs)

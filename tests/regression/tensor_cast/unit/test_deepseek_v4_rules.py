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

"""Pure routing and sparse-topk rules for DeepSeek V4 components."""

from unittest.mock import MagicMock

import torch.nn as nn
from tensor_cast.layers.deepseek_v4 import (
    get_compress_topk_idxs,
    get_window_topk_idxs,
    has_deepseek_v4_hash_routing,
)
from tensor_cast.layers.mla import _resolve_sparse_topk_limit


def test_hash_routing_prefers_explicit_gate_attribute():
    gate = nn.Linear(4, 2)
    gate.hash = True
    assert has_deepseek_v4_hash_routing(gate, moe_layer_idx=None) is True


def test_hash_routing_explicit_false_is_preserved():
    gate = nn.Linear(4, 2)
    gate.hash = False
    assert has_deepseek_v4_hash_routing(gate, moe_layer_idx=None) is False


def test_hash_routing_falls_back_to_early_moe_layer_index():
    gate = nn.Linear(4, 2)
    assert has_deepseek_v4_hash_routing(gate, moe_layer_idx=2) is True
    assert has_deepseek_v4_hash_routing(gate, moe_layer_idx=3) is False
    assert has_deepseek_v4_hash_routing(gate, moe_layer_idx=0) is True


def test_window_topk_shape_for_prefill():
    assert get_window_topk_idxs(128, 2, 32, "cpu", is_decode=False).shape == (2, 32, 32)


def test_window_topk_shape_for_decode():
    assert get_window_topk_idxs(128, 2, 1, "cpu", is_decode=True).shape == (2, 1, 128)


def test_compress_topk_shape():
    assert get_compress_topk_idxs(4, 2, 32, "cpu").shape == (2, 32, 8)


def test_compress_topk_uses_total_sequence_length():
    assert get_compress_topk_idxs(128, 2, 32, "cpu", total_seq_length=1024).shape == (2, 32, 8)


def test_sparse_topk_resolves_explicit_value():
    assert _resolve_sparse_topk_limit(MagicMock(), topk_limit=16) == 16


def test_sparse_topk_resolves_inner_value():
    inner = MagicMock(topk_limit=24)
    assert _resolve_sparse_topk_limit(inner) == 24


def test_sparse_topk_resolves_config_value():
    inner = MagicMock(topk_limit=None, config=MagicMock(topk_limit=32))
    assert _resolve_sparse_topk_limit(inner) == 32


def test_sparse_topk_falls_back_to_index_topk():
    inner = MagicMock(topk_limit=None, config=MagicMock(topk_limit=None, index_topk=48))
    assert _resolve_sparse_topk_limit(inner) == 48

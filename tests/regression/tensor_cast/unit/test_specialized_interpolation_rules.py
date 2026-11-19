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

"""Pure axes, workload, routing, and scale-mode rules for specialized interpolation."""

import pytest
from tensor_cast.performance_model.profiling_database.interpolating_data_source import InterpolatingDataSource


def test_elementwise_axes_use_total_input_output_numel():
    assert InterpolatingDataSource._elementwise_axes_from_shapes([(2, 3), (1, 3)], (2, 3)) == {"io_numel": 15.0}


def test_elementwise_axes_reject_negative_dimensions():
    assert InterpolatingDataSource._elementwise_axes_from_shapes([(-1, 3)], (2, 3)) is None
    assert InterpolatingDataSource._elementwise_axes_from_shapes([(2, 3)], (-1, 3)) is None


def test_attention_runtime_workload_uses_query_weighted_kv_length():
    assert InterpolatingDataSource._attention_runtime_workload(q_tokens=6, query_lengths=(2, 4), kv_lengths=(2, 8)) == {
        "q_tokens": 6.0,
        "effective_kv_len": 6.0,
        "phase": "mixed",
        "batch_size": 2,
    }


def test_attention_runtime_workload_rejects_negative_kv_length():
    assert (
        InterpolatingDataSource._attention_runtime_workload(q_tokens=2, query_lengths=(2, 0), kv_lengths=(4, -1))
        is None
    )


def test_runtime_attention_regime_rejects_malformed_numeric_fields():
    params = {
        "sparse_mode": 3,
        "num_kv_heads": 1,
        "input_layout": "TND",
        "topk": "invalid",
        "block_size": 128,
        "num_heads": 32,
        "cache_layout": "PA_BSND",
        "kv_cache_mode": "paged",
    }
    assert (
        InterpolatingDataSource._runtime_attention_regime_fields(
            "LightningIndexer", "DT_BF16", (1, 32, 128), {"phase": "decode"}, params, include_sparse_fields=False
        )
        is None
    )


def test_query_lengths_from_cumulative_offsets_fails_closed():
    assert InterpolatingDataSource._query_lengths_from_cumulative(6, (2, 6)) == (2, 4)
    assert InterpolatingDataSource._query_lengths_from_cumulative(6, (2, 5)) is None
    assert InterpolatingDataSource._query_lengths_from_cumulative(6, (3, 2, 6)) is None


@pytest.mark.parametrize(
    ("shapes", "dtypes", "expected"),
    [
        (
            [(2, 4, 72, 7168), (16, 7168, 4096), (16, 2048, 7168), (2, 4, 72, 4)],
            ["DT_BF16", "INT8", "INT8", "INT32"],
            4,
        ),
        (
            [(1, 180, 7168), (16, 7168, 4096), (16, 2048, 7168), (180, 4), (180, 8)],
            ["DT_BF16", "INT8", "INT8", "INT32", "INT64"],
            4,
        ),
        ([(1, 180, 7168), (16, 7168, 4096), (16, 2048, 7168), (180, 4)], ["DT_BF16", "INT8", "INT8", "UINT32"], None),
    ],
)
def test_moe_fused_topk_uses_discrete_integer_route_input(shapes, dtypes, expected):
    assert (
        InterpolatingDataSource._moe_fused_topk(
            shapes, tokens=576.0 if len(shapes[0]) == 4 else 180.0, input_dtypes=dtypes
        )
        == expected
    )


def test_compute_scale_mode_keeps_block_quant_kernel_specific():
    mode = InterpolatingDataSource._compute_scale_mode
    assert mode((8, 64), (), "DynamicQuant") == ("per_tensor", None)
    assert mode((8, 64), (8, 1), "DynamicQuant") == ("per_token", None)
    assert mode((8, 64), (1, 64), "DynamicQuant") == ("per_channel", None)
    assert mode((8, 64), (4,), "DynamicQuant") is None
    assert mode((8, 64), (4,), "DynamicBlockQuant") == ("per_block", 16)

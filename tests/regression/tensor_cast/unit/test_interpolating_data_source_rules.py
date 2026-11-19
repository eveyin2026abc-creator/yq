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

"""Pure candidate, tensor-slot, and packed-shape rules for interpolation."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch
from tensor_cast.performance_model.profiling_database.interpolating_data_source import InterpolatingDataSource
from tensor_cast.performance_model.profiling_database.profiling_data_source import ProfilingDataSource


def _op(input_tensors):
    op = MagicMock()
    op.args = tuple(input_tensors)
    op.kwargs = {}
    op.out = None
    return op


@pytest.mark.parametrize(
    ("row", "expected", "column"),
    [
        ({"Average Duration(us)": 0.0, "Profiling Average Duration(us)": 12.5}, 12.5, "Profiling Average Duration(us)"),
        (
            {
                "Average Duration(us)": 0.0,
                "Profiling Average Duration(us)": "bad",
                "Profiling Median Duration(us)": 12.0,
                "Duration(us)": 99.0,
            },
            12.0,
            "Profiling Median Duration(us)",
        ),
    ],
)
def test_candidate_latency_uses_alternate_valid_column(row, expected, column):
    latency, meta = object.__new__(InterpolatingDataSource)._candidate_latency(pd.Series(row), "Average Duration(us)")
    assert latency == pytest.approx(expected)
    assert meta["latency_column"] == column
    assert meta["latency_column_selection"] == "alternate_latency_column"


def test_candidate_latency_rejects_non_finite_and_non_positive_cells():
    row = pd.Series(
        {
            "Average Duration(us)": float("nan"),
            "Profiling Average Duration(us)": float("inf"),
            "Profiling Median Duration(us)": -1.0,
            "Median Duration(us)": 0.0,
        }
    )
    latency, meta = object.__new__(InterpolatingDataSource)._candidate_latency(row, "Average Duration(us)")
    assert latency is None
    assert meta["latency_rejected_reason"] == "latency_invalid"


def test_base_row_latency_pair_keeps_priority_and_skips_zero_primary_column():
    row = pd.Series({"Average Duration(us)": 0.0, "Profiling Average Duration(us)": 12.5, "Duration(us)": 99.0})
    assert ProfilingDataSource._row_latency_pair(row, "Average Duration(us)") == (
        "Profiling Average Duration(us)",
        12.5,
    )


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([torch.zeros((4, 8)), torch.zeros((8, 16))], [((4, 8), torch.float32), ((8, 16), torch.float32)]),
        ([[torch.zeros((4, 8))]], [((4, 8), torch.float32)]),
        ([[torch.zeros((3, 8)), torch.zeros((5, 8))]], [((8, 8), torch.float32)]),
        (
            [[torch.zeros((3, 8)), torch.zeros((5, 8))], [torch.zeros((8, 16)), torch.zeros((8, 16))]],
            [((8, 8), torch.float32), ((2, 8, 16), torch.float32)],
        ),
        (
            [torch.zeros((4, 8)), 3, "x", None, torch.zeros((8, 16))],
            [((4, 8), torch.float32), ((8, 16), torch.float32)],
        ),
        ([torch.tensor(1.0), torch.zeros((8, 16))], [((8, 16), torch.float32)]),
    ],
)
def test_extract_slot_tensor_inputs(args, expected):
    assert InterpolatingDataSource._extract_slot_tensor_inputs(_op(args)) == expected


@pytest.mark.parametrize(
    ("mapping", "shapes", "dtypes", "expected"),
    [
        ({}, [(16, 64), (32, 128)], [torch.bfloat16, torch.int8], [(16, 64), (32, 128)]),
        (
            {"expected_input_dtypes": ["DT_BF16", "DT_INT4"]},
            [(16, 64), (32, 128)],
            [torch.bfloat16, torch.int8],
            [(16, 64), (64, 128)],
        ),
        (
            {"expected_input_dtypes": ["DT_BF16", "DT_INT4"]},
            [(16, 64), (128, 32)],
            [torch.bfloat16, torch.int8],
            [(16, 64), (128, 64)],
        ),
        (
            {"expected_input_dtypes": ["DT_BF16", "DT_INT4"]},
            [(16, 64), (8, 128)],
            [torch.bfloat16, torch.int32],
            [(16, 64), (64, 128)],
        ),
        (
            {"expected_input_dtypes": ["DT_BF16", "DT_BF16", "DT_INT4"]},
            [(16, 64), (16, 64), (33, 128)],
            [torch.bfloat16, torch.bfloat16, torch.int8],
            [(16, 64), (16, 64), (33, 128)],
        ),
    ],
)
def test_restore_packed_int4_weight_shapes(mapping, shapes, dtypes, expected):
    assert InterpolatingDataSource._restore_packed_int4_weight_shapes(mapping, shapes, dtypes) == expected

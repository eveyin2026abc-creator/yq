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

"""Pure topology, dispatch-overhead, and iteration rules for communication microbench."""

import csv

import pytest
from tools.perf_data_collection.comm_bench.generate_comm_microbench import (
    _DISPATCH_OVERHEAD,
    PROFILER_ACTIVE_ITERS,
    PROFILER_ACTIVE_ITERS_LARGE,
    PROFILER_LARGE_MSG_THRESHOLD,
    _active_iters_for_msg,
    _apply_dispatch_overhead,
    _parse_kernel_comm_duration,
    build_group_for_tier,
    resolve_topology_tier,
)

GRID = [48, 8, 2]


@pytest.mark.parametrize(
    ("ranks", "expected"),
    [
        (list(range(16)), 1),
        (list(range(8)), 1),
        ([0, 1], 2),
        ([0, 1, 2, 3], 1),
        (list(range(128)), 0),
        ([5], 2),
    ],
)
def test_resolve_topology_tier(ranks, expected):
    assert resolve_topology_tier(ranks, GRID) == expected


@pytest.mark.parametrize(
    ("num_devices", "tier", "expected"),
    [(16, 1, list(range(16))), (8, 1, list(range(8))), (2, 2, [0, 1])],
)
def test_build_group_for_tier(num_devices, tier, expected):
    assert build_group_for_tier(0, num_devices, tier, GRID) == expected


def test_group_resolves_back_to_same_tier():
    for num_devices, tier in [(16, 1), (8, 1), (2, 2)]:
        assert resolve_topology_tier(build_group_for_tier(0, num_devices, tier, GRID), GRID) == tier


def test_build_group_rejects_size_above_tier_span():
    with pytest.raises(ValueError, match="exceeds span size"):
        build_group_for_tier(0, 4, 2, GRID)


def _row(op_type, num_devices, duration_us, message_bytes=1048576):
    return {
        "message_bytes": message_bytes,
        "num_devices": num_devices,
        "dtype": "DT_BF16",
        "topology_tier": 1,
        "Duration(us)": duration_us,
        "bandwidth_gbps": round(message_bytes / (duration_us * 1e-6) / 1e9, 2),
    }


def test_all_known_dispatch_overheads_are_applied():
    for (op_type, num_devices), overhead in _DISPATCH_OVERHEAD.items():
        result = _apply_dispatch_overhead(_row(op_type, num_devices, 100.0), op_type)
        assert result["Duration(us)"] == pytest.approx(100.0 + overhead, abs=0.01)


def test_bandwidth_is_recalculated_after_overhead():
    result = _apply_dispatch_overhead(_row("all_gather", 16, 100.0), "all_gather")
    duration = 100.0 + _DISPATCH_OVERHEAD[("all_gather", 16)]
    assert result["bandwidth_gbps"] == pytest.approx(round(1048576 / (duration * 1e-6) / 1e9, 2), abs=0.01)


def test_missing_operation_overhead_returns_original_row():
    row = _row("all_to_all", 16, 100.0)
    assert _apply_dispatch_overhead(row, "all_to_all") is row


def test_unknown_device_count_returns_original_row():
    row = _row("all_reduce", 4, 100.0)
    assert _apply_dispatch_overhead(row, "all_reduce") is row


def test_dispatch_overhead_does_not_mutate_original_row():
    row = _row("all_reduce", 16, 100.0)
    _apply_dispatch_overhead(row, "all_reduce")
    assert row["Duration(us)"] == 100.0


@pytest.mark.parametrize(
    ("message_bytes", "expected"),
    [
        (65536, PROFILER_ACTIVE_ITERS),
        (PROFILER_LARGE_MSG_THRESHOLD - 1, PROFILER_ACTIVE_ITERS),
        (PROFILER_LARGE_MSG_THRESHOLD, PROFILER_ACTIVE_ITERS_LARGE),
        (PROFILER_LARGE_MSG_THRESHOLD * 4, PROFILER_ACTIVE_ITERS_LARGE),
    ],
)
def test_active_iterations_follow_message_threshold(message_bytes, expected):
    assert _active_iters_for_msg(message_bytes) == expected


def _write_kernel_details(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Type", "Name", "Duration(us)"])
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (
            [
                {"Type": "hcom_allReduce_", "Name": "hcom_allReduce_0", "Duration(us)": "100.0"},
                {"Type": "hcom_allReduce_", "Name": "hcom_allReduce_1", "Duration(us)": "120.0"},
            ],
            [100.0, 120.0],
        ),
        (
            [
                {"Type": "hcom_allReduce_", "Name": "hcom_allReduce_0", "Duration(us)": "100.0"},
                {"Type": "hcom_allReduce_", "Name": "AivKernel_hcom_allReduce_", "Duration(us)": "100.0"},
            ],
            [100.0],
        ),
        (
            [
                {"Type": "hcom_allReduce_", "Name": "hcom_allReduce_0", "Duration(us)": "0"},
                {"Type": "hcom_allReduce_", "Name": "hcom_allReduce_1", "Duration(us)": "150.0"},
            ],
            [150.0],
        ),
        (
            [
                {"Type": "hcom_allGather_", "Name": "hcom_allGather_0", "Duration(us)": "200.0"},
                {"Type": "hcom_allReduce_", "Name": "hcom_allReduce_0", "Duration(us)": "100.0"},
            ],
            [100.0],
        ),
    ],
)
def test_parse_kernel_comm_duration_filters_profiler_rows(tmp_path, rows, expected):
    _write_kernel_details(tmp_path / "kernel_details.csv", rows)
    assert _parse_kernel_comm_duration(str(tmp_path), "all_reduce") == expected


def test_parse_kernel_comm_duration_returns_empty_for_missing_csv(tmp_path):
    assert _parse_kernel_comm_duration(str(tmp_path), "all_reduce") == []


def test_parse_kernel_comm_duration_finds_nested_csv(tmp_path):
    nested = tmp_path / "rank0" / "profiler_output"
    nested.mkdir(parents=True)
    _write_kernel_details(
        nested / "kernel_details.csv",
        [{"Type": "hcom_allReduce_", "Name": "hcom_allReduce_0", "Duration(us)": "80.0"}],
    )
    assert _parse_kernel_comm_duration(str(tmp_path), "all_reduce") == [80.0]

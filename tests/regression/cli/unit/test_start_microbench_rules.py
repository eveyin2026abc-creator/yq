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
"""Pure formatting and shard-option rules for the microbench CLI."""

import importlib
import sys
from pathlib import Path

import pytest

PERF_DATA_COLLECTION_DIR = Path(__file__).resolve().parents[4] / "tools" / "perf_data_collection"
if str(PERF_DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(PERF_DATA_COLLECTION_DIR))

start_microbench = importlib.import_module("start_microbench")
md_table = start_microbench.md_table
validate_case_shard_options = start_microbench.validate_case_shard_options
get_cols = start_microbench.get_cols
should_skip_dispatch_ffn_msprof = start_microbench.should_skip_dispatch_ffn_msprof


class TestMdTable:
    def test_empty_rows_returns_none(self):
        assert md_table(["Col1", "Col2"], []) == "_None_"

    def test_single_row(self):
        lines = md_table(["Name", "Value"], [["foo", "bar"]]).split("\n")
        assert "Name" in lines[0]
        assert "foo" in lines[2]
        assert "bar" in lines[2]
        assert "---" in lines[1]

    def test_multiple_rows(self):
        lines = md_table(["Op", "Count"], [["Add", "5"], ["MatMul", "3"]]).split("\n")
        assert "Op" in lines[0]
        assert "Add" in lines[2]
        assert "MatMul" in lines[3]
        assert "5" in lines[2]

    def test_column_width_alignment(self):
        lines = md_table(["Name", "Value"], [["a", "x"], ["longer_name", "y"]]).split("\n")
        assert len(lines) == 4


class TestCaseShardOptions:
    def test_rejects_pruning_during_multi_shard_collection(self):
        with pytest.raises(ValueError, match="cannot be combined with multi-shard replay"):
            validate_case_shard_options(4, 0, prune_empty_duration_rows=True)

    def test_accepts_multi_shard_collection_without_pruning(self):
        validate_case_shard_options(4, 3, prune_empty_duration_rows=False)

    def test_rejects_out_of_range_shard_index(self):
        with pytest.raises(ValueError, match="case shard index"):
            validate_case_shard_options(4, 4, prune_empty_duration_rows=False)


class TestDispatchFfnSkip:
    @pytest.mark.parametrize(
        ("selected_ops", "ep_size", "local_ranks", "visible_devices", "expected"),
        [
            (["DispatchFFNCombine"], 16, None, 2, True),
            (None, 16, None, 2, True),
            (["DispatchFFNCombine"], 32, 16, 16, False),
            (["DispatchFFNCombine", "MatMulV2"], 16, None, 2, False),
        ],
    )
    def test_dispatch_ffn_skip_rule(self, selected_ops, ep_size, local_ranks, visible_devices, expected):
        assert (
            should_skip_dispatch_ffn_msprof(
                selected_ops,
                ep_size=ep_size,
                nproc_per_node=local_ranks,
                visible_devices=visible_devices,
                update_mode="missing-only",
                has_prof_path=False,
            )
            is expected
        )


class TestGetCols:
    def test_none_returns_full_schema(self):
        cols = get_cols(None)
        assert len(cols) == 62
        assert "Average Duration(us)" in cols
        assert "Profiling Average Duration(us)" in cols
        assert "Profiling Median Duration(us)" in cols
        assert "Profiling Std Duration(us)" in cols
        assert "MicroBench aicore_time(us)" in cols

    def test_excludes_legacy_columns(self):
        cols = get_cols(
            [
                "OP State",
                "Input Shapes",
                "Average Duration(us)",
                "MicroBench Task Duration(us)",
                "MicroBench Kernel Duration(us)",
            ]
        )
        assert "MicroBench Task Duration(us)" not in cols
        assert "MicroBench Kernel Duration(us)" not in cols

    def test_converts_legacy_mb_dur_to_new(self):
        cols = get_cols(["OP State", "Input Shapes", "MicroBench Duration(us)"])
        assert "Average Duration(us)" in cols
        assert "MicroBench Duration(us)" not in cols

    def test_inserts_mb_cols_before_profiling_cols(self):
        cols = get_cols(["OP State", "Input Shapes", "Average Duration(us)", "Profiling Average aicore_time(us)"])
        assert cols.index("MicroBench aicore_time(us)") < cols.index("Profiling Average aicore_time(us)")

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
"""Pure query-runner discovery, cache-key, and input-normalization rules."""

from pathlib import Path

import pytest

from tools.perf_data_collection.grid_generator.runner import (
    _query_cache_directory,
    discover_replay_supported_ops,
    iter_csv_files,
    load_csv_files,
    normalize_selected_ops,
    normalize_target_models,
)


class TestCsvDiscovery:
    def test_sorts_and_excludes_tmp_files(self, tmp_path: Path) -> None:
        (tmp_path / "PadV3.csv").write_text("a", encoding="utf-8")
        (tmp_path / "MatMulV2.csv").write_text("b", encoding="utf-8")
        (tmp_path / "MatMulV2.tmp.csv").write_text("c", encoding="utf-8")
        assert [path.name for path in iter_csv_files(tmp_path)] == ["MatMulV2.csv", "PadV3.csv"]

    def test_load_rejects_empty_or_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No CSV files"):
            load_csv_files(tmp_path)
        with pytest.raises(ValueError, match="does not exist"):
            load_csv_files(tmp_path / "missing")

    def test_query_cache_is_invalidated_by_database_or_model_change(self, tmp_path: Path) -> None:
        database = tmp_path / "database"
        database.mkdir()
        (database / "op_mapping.yaml").write_text("device: TEST\n", encoding="utf-8")
        csv_path = database / "Add.csv"
        csv_path.write_text("Input Shapes\n1,4\n", encoding="utf-8")

        original = _query_cache_directory(database, ["org/model"], tmp_path)
        other_model = _query_cache_directory(database, ["org/other"], tmp_path)
        csv_path.write_text("Input Shapes\n2,4\n", encoding="utf-8")
        changed_database = _query_cache_directory(database, ["org/model"], tmp_path)

        assert original != other_model
        assert original != changed_database


class TestPublicInputNormalization:
    def test_target_models_accept_spaces_and_commas(self) -> None:
        assert normalize_target_models(["org/a,org/b", "org/a"]) == ["org/a", "org/b"]

    def test_replay_support_is_discovered_from_scripts(self, tmp_path: Path) -> None:
        (tmp_path / "Add_run.py").write_text("", encoding="utf-8")
        (tmp_path / "RmsNorm_run.py").write_text("", encoding="utf-8")
        (tmp_path / "helper.py").write_text("", encoding="utf-8")
        assert discover_replay_supported_ops(tmp_path) == ("Add", "RmsNorm")

    def test_selected_ops_fail_closed_outside_replay_support(self) -> None:
        with pytest.raises(ValueError, match="without an op_replay"):
            normalize_selected_ops(["Unknown"], ["Add"])

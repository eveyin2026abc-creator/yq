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
"""Tests for nightly core/cov_helper coverage separation."""

from pathlib import Path

from coverage import CoverageData

from scripts.helpers.nightly.coverage_split import split_line_coverage


def test_split_line_coverage_reports_helper_only_increment(tmp_path: Path) -> None:
    source = tmp_path / "product.py"
    source.write_text("value = 1\nother = 2\n", encoding="utf-8")
    data_file = tmp_path / ".coverage"
    data = CoverageData(basename=str(data_file))
    data.set_context("tests/test_product.py::test_core|run")
    data.add_lines({str(source): {1}})
    data.set_context("tests/test_product.py::test_helper|run")
    data.add_lines({str(source): {2}})
    data.write()

    result = split_line_coverage(data_file, {"tests/test_product.py::test_helper"})

    assert result.core_line_percent == 50.0
    assert result.cov_helper_line_contribution == 50.0
    assert result.cov_helper_only_lines == 1

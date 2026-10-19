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

"""Tests for the coverage SQLite durability plugin."""

from __future__ import annotations

from pathlib import Path

from coverage.debug import NoDebugging
from coverage.sqlitedb import SqliteDb

from scripts.helpers.common import coverage_sqlite


def test_coverage_sqlite_plugin_uses_delete_journal(tmp_path: Path) -> None:
    coverage_sqlite._enable_durable_coverage_sqlite()
    data_file = tmp_path / ".coverage"
    with SqliteDb(str(data_file), NoDebugging()) as database:
        database.execute_void("create table sample (value integer)")
        database.execute_void("insert into sample values (1)")
        with database:
            assert database.execute_one("select value from sample") == (1,)
        journal = database.execute_one("pragma journal_mode")
        synchronous = database.execute_one("pragma synchronous")
    assert journal == ("delete",)
    assert synchronous == (1,)

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

"""Pytest plugin that keeps coverage SQLite files durable under pytest-xdist.

coverage.py writes data files with ``PRAGMA journal_mode=off``. A torn write can
leave ``coverage_schema`` in place while ``context`` is missing, and the worker
then dies in ``coverage.save()``. DELETE journaling keeps the parallel workers.
"""

from __future__ import annotations

import functools
from typing import Any


def _enable_durable_coverage_sqlite() -> None:
    from coverage.sqlitedb import SqliteDb

    original = SqliteDb._connect
    if getattr(original, "_msmodeling_durable_sqlite", False):
        return

    @functools.wraps(original)
    def _connect(self: Any) -> None:
        was_connected = self.con is not None
        original(self)
        if was_connected or self.con is None:
            return
        self.execute_void("pragma journal_mode=DELETE")
        self.execute_void("pragma synchronous=NORMAL")

    _connect._msmodeling_durable_sqlite = True  # type: ignore[attr-defined]
    SqliteDb._connect = _connect  # type: ignore[method-assign]


# Explicit ``-p`` imports this module before pytest-cov starts collection.
_enable_durable_coverage_sqlite()

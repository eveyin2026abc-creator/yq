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

"""Pytest plugin that prints collected node ids, one per line."""

from __future__ import annotations

from typing import Protocol


class _Node(Protocol):
    nodeid: str


class _Session(Protocol):
    items: list[_Node]


def pytest_collection_finish(session: _Session) -> None:
    """Print each collected node id. ``-qq`` itself only prints ``path: count``."""
    for item in session.items:
        print(item.nodeid, flush=True)

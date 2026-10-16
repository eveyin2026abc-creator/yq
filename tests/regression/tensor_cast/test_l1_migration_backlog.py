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

"""The backlog names every direct construction the census still finds."""

from collections import Counter

from tests.helpers.construction_census import iter_sites
from tests.helpers.l1_migration_backlog import DIRECT_SITES


def test_backlog_matches_the_census_and_keeps_the_executor():
    counted = Counter(
        (site.path, site.function, site.callee)
        for site in iter_sites()
        if site.direct_model
    )
    listed = {(note.path, note.function, note.callee): note for note in DIRECT_SITES}
    assert set(listed) == set(counted)
    for key, note in listed.items():
        assert note.calls == counted[key]
        assert note.disposition in {"keep", "later"}
        assert note.reason
    executor = listed[("tests/helpers/l1_scenario.py", "runner", "ModelRunner")]
    assert executor.disposition == "keep"
    assert any(note.disposition == "later" for note in DIRECT_SITES)

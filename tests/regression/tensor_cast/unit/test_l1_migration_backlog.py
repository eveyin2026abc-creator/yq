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

import csv
from collections import Counter
from pathlib import Path

from tests.architecture.construction_census import iter_sites

_BACKLOG = Path(__file__).resolve().parents[3] / ".ci/direct_construction_backlog.csv"


def _backlog_rows() -> tuple[dict[str, str], ...]:
    with _BACKLOG.open(newline="", encoding="utf-8") as stream:
        return tuple(csv.DictReader(stream))


def test_backlog_matches_the_census_and_keeps_the_executor():
    counted = Counter((site.path, site.function, site.callee) for site in iter_sites() if site.direct_model)
    rows = _backlog_rows()
    listed = {(row["path"], row["function"], row["callee"]): row for row in rows}
    assert set(listed) == set(counted)
    for key, row in listed.items():
        assert int(row["calls"]) == counted[key]
        assert row["disposition"] in {"keep", "later"}
        assert row["reason"]
    executor = listed[("tests/helpers/l1_scenario.py", "runner", "ModelRunner")]
    assert executor["disposition"] == "keep"
    assert any(row["disposition"] == "later" for row in rows)

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

"""Every L2 catalog row builds, or fails with the error the row names."""

import pytest

from tests.helpers.l2_catalog import CATALOG, catalog_covers_kinds, run_catalog_case


def test_catalog_covers_the_seven_kinds_and_rejects_a_model_identity():
    assert catalog_covers_kinds()
    rejected = [case for case in CATALOG if not case.valid and case.name == "rejects_model_id"]
    assert len(rejected) == 7


@pytest.mark.parametrize("case", CATALOG, ids=lambda case: f"{case.kind}-{case.name}")
def test_catalog_case(case):
    if case.valid:
        artifact = run_catalog_case(case)
        assert artifact.kind == case.kind
        return
    with pytest.raises((TypeError, ValueError), match=case.error):
        run_catalog_case(case)

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

"""L3 contracts cover exit codes, exceptions, and defaults without a model."""

import pytest

from tests.helpers.l3_contracts import require_default, require_exit_code, require_raises


def _exit_with_code_2() -> None:
    raise SystemExit(2)


def test_require_exit_code_returns_the_matching_code():
    assert require_exit_code(_exit_with_code_2, 2) == 2


def test_require_exit_code_rejects_a_normal_return():
    with pytest.raises(AssertionError, match="returned"):
        require_exit_code(lambda: None, 2)


def _raise_bad_device() -> None:
    raise ValueError("bad device")


def test_require_raises_returns_the_exception():
    exc = require_raises(_raise_bad_device, ValueError)
    assert str(exc) == "bad device"


def test_require_default_fills_none_and_rejects_the_wrong_type():
    assert require_default(None, 1, "num_queries") == 1
    assert require_default(4, 1, "num_queries") == 4
    with pytest.raises(TypeError, match="num_queries"):
        require_default("4", 1, "num_queries")

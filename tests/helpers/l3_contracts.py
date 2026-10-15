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

"""Pure L3 contracts: exit codes, exceptions, and defaults."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def require_exit_code(action: Callable[[], None], code: int) -> int:
    """Run *action* and return the ``SystemExit`` code, which must equal *code*."""
    try:
        action()
    except SystemExit as exc:
        actual = exc.code if isinstance(exc.code, int) else 1
        if actual != code:
            raise AssertionError(f"exit code {actual} != {code}") from exc
        return actual
    raise AssertionError(f"action returned instead of exiting with code {code}")


def require_raises(action: Callable[[], object], exc_type: type[BaseException]) -> BaseException:
    """Run *action* and return the expected exception."""
    try:
        action()
    except exc_type as exc:
        return exc
    except Exception as exc:
        raise AssertionError(f"raised {type(exc).__name__}, expected {exc_type.__name__}") from exc
    raise AssertionError(f"action returned instead of raising {exc_type.__name__}")


def require_default(value: T | None, default: T, name: str) -> T:
    """Return *value* when it is set, otherwise *default*."""
    if value is None:
        return default
    if type(value) is not type(default):
        raise TypeError(f"{name} must be {type(default).__name__}, got {type(value).__name__}")
    return value

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

"""The field-role table stays aligned with UserInputConfig and the executor."""

from dataclasses import fields

from tensor_cast.core.user_config import UserInputConfig

from tests.helpers.l1_field_roles import FIELD_NOTES, field_role
from tests.helpers.l1_scenario import OBSERVABILITY_FIELDS, RUN_FIELDS, build_field_names


def test_every_user_input_field_has_one_note_and_one_role():
    names = [config_field.name for config_field in fields(UserInputConfig)]
    assert set(FIELD_NOTES) == set(names)
    assert len(FIELD_NOTES) == len(names)
    for name in names:
        assert FIELD_NOTES[name].strip()
        role = field_role(name)
        if name in RUN_FIELDS:
            assert role == "run"
        elif name in OBSERVABILITY_FIELDS:
            assert role == "observability"
        else:
            assert role == "build"


def test_build_fields_are_the_ones_outside_run_and_observability():
    classified = {name for name, _note in FIELD_NOTES.items() if field_role(name) == "build"}
    assert classified == set(build_field_names())

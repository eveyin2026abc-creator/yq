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

"""Build-signature probes for every vendored config. No model is constructed."""

from pathlib import Path

from tests.helpers.l1_case_matrix import (
    FIXTURE_ROWS,
    build_probes,
    read_model_type,
    representative_matches_family_table,
    representatives,
    rows_for_stack,
    same_build,
)

_REPO = Path(__file__).resolve().parents[3]


def test_representatives_match_the_family_table():
    assert representative_matches_family_table()
    assert len(representatives()) == 18


def test_every_fixture_config_exists_and_has_a_model_type():
    stacks = {row.stack for row in representatives()}
    for row in FIXTURE_ROWS:
        config = _REPO / row.model_id / "config.json"
        assert config.is_file(), row.model_id
        assert read_model_type(row.model_id)
        if row.role == "sibling":
            assert row.stack in stacks


def test_sibling_checkpoint_does_not_share_a_build_with_its_representative():
    by_stack = {row.stack: row for row in representatives()}
    for row in FIXTURE_ROWS:
        if row.role != "sibling":
            continue
        representative = by_stack[row.stack]
        assert row.model_id != representative.model_id
        assert not same_build(
            build_probes(row)[0].left,
            build_probes(representative)[0].left,
        )


def test_reuse_probes_hold_for_every_fixture():
    for row in FIXTURE_ROWS:
        probes = build_probes(row)
        names = [probe.name for probe in probes]
        assert names == [
            "query_len",
            "num_queries",
            "log_level",
            "quantization",
            "layers",
            "compile",
            "context_length",
            "device",
        ]
        for probe in probes:
            assert same_build(probe.left, probe.right) is probe.same_build


def test_dense_sizes_share_one_stack():
    dense = rows_for_stack("qwen3_dense")
    assert len(dense) == 6
    assert sum(row.role == "representative" for row in dense) == 1

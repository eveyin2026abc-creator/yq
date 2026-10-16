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

"""Local model fixtures and the L1 build-reuse probes that apply to each one.

One representative per architecture family is enough for a real build. Sibling
checkpoints of the same family are listed so a later migration does not open a
second build row for every size. This module only builds ``UserInputConfig``
values. It does not construct ``ModelRunner``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig

from tests.helpers.l1_families import FAMILY_MODEL_IDS
from tests.helpers.l1_scenario import L1BuildSignature

_REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FixtureRow:
    """One vendored config and the reuse row it belongs to."""

    name: str
    model_id: str
    stack: str
    role: str
    vision: bool


# role is one of: representative, sibling, smoke, remote, outside, network.
FIXTURE_ROWS: tuple[FixtureRow, ...] = (
    FixtureRow("qwen3_dense", "tests/assets/model_config/qwen3_dense_0_6b", "qwen3_dense", "representative", False),
    FixtureRow("qwen3_dense_1_7b", "tests/assets/model_config/qwen3_dense_1_7b", "qwen3_dense", "sibling", False),
    FixtureRow("qwen3_dense_4b", "tests/assets/model_config/qwen3_dense_4b", "qwen3_dense", "sibling", False),
    FixtureRow("qwen3_dense_8b", "tests/assets/model_config/qwen3_dense_8b", "qwen3_dense", "sibling", False),
    FixtureRow("qwen3_dense_14b", "tests/assets/model_config/qwen3_dense_14b", "qwen3_dense", "sibling", False),
    FixtureRow("qwen3_dense_32b", "tests/assets/model_config/qwen3_dense_32b", "qwen3_dense", "sibling", False),
    FixtureRow("qwen3_moe", "tests/assets/model_config/qwen3_moe_30b_a3b", "qwen3_moe", "representative", False),
    FixtureRow("qwen3_moe_235b", "tests/assets/model_config/qwen3_moe_235b_a22b", "qwen3_moe", "sibling", False),
    FixtureRow("qwen3_vl", "tests/assets/model_config/qwen3_vl_8b_instruct", "qwen3_vl", "representative", True),
    FixtureRow("qwen3_vl_tiny", "tests/assets/model_config/qwen3_vl_tiny", "qwen3_vl", "sibling", True),
    FixtureRow("qwen3_vl_moe", "tests/assets/model_config/qwen3_vl_moe_235b_a22b", "qwen3_vl_moe", "representative", True),
    FixtureRow("qwen3_5", "tests/assets/model_config/qwen3_5_27b", "qwen3_5", "representative", False),
    FixtureRow("qwen3_5_moe", "tests/assets/model_config/qwen3_5_moe_397b_a17b", "qwen3_5_moe", "representative", False),
    FixtureRow("qwen3_next", "tests/assets/model_config/qwen3_next_80b_a3b", "qwen3_next", "representative", False),
    FixtureRow("deepseek_v3", "tests/assets/model_config/deepseek_v3", "deepseek_v3", "representative", False),
    FixtureRow("deepseek_v31", "tests/assets/model_config/deepseek_v31", "deepseek_v3", "sibling", False),
    FixtureRow("deepseek_v32", "tests/assets/model_config/deepseek_v32", "deepseek_v32", "representative", False),
    FixtureRow("deepseek_v4", "tests/assets/model_config/deepseek_v4_flash", "deepseek_v4", "representative", False),
    FixtureRow("deepseek_new", "tests/assets/model_config/deepseek_new", "deepseek_v3", "smoke", False),
    FixtureRow("deepseek_v31_remote", "tests/assets/model_config/deepseekv3.1_remote", "deepseek_v3", "remote", False),
    FixtureRow(
        "deepseek_v31_remote_json",
        "tests/assets/model_config/deepseekv3.1_remote_json_only",
        "deepseek_v3",
        "remote",
        False,
    ),
    FixtureRow("glm5", "tests/assets/model_config/glm5_1", "glm5", "representative", False),
    FixtureRow("glm5_base", "tests/assets/model_config/glm5", "glm5", "sibling", False),
    FixtureRow("glm4", "tests/assets/model_config/glm4_7", "glm4", "representative", False),
    FixtureRow("glm4v", "tests/assets/model_config/glm4v_9b_thinking", "glm4v", "representative", True),
    FixtureRow("glm4v_moe", "tests/assets/model_config/glm4v_moe_4_5v", "glm4v_moe", "representative", True),
    FixtureRow("minimax_m2", "tests/assets/model_config/minimax_m2", "minimax_m2", "representative", False),
    FixtureRow("minimax_m2_7", "tests/assets/model_config/minimax_m2_7", "minimax_m2", "sibling", False),
    FixtureRow("mimo_v2", "tests/assets/model_config/mimo_v2_flash", "mimo_v2", "representative", False),
    FixtureRow("ling", "tests/assets/model_config/ling_flash_2_0", "ling", "representative", False),
    FixtureRow("kimi_k25", "tests/assets/model_config/kimi_k2_5", "kimi_k25", "representative", True),
    FixtureRow("qwen2_5", "tests/assets/model_config/qwen2_5_7b", "qwen2_5", "outside", False),
    FixtureRow("kimi_k2_thinking", "tests/assets/model_config/kimi_k2_thinking", "kimi_k2", "network", False),
)


@dataclass(frozen=True)
class BuildProbe:
    """Two configs and whether they must share one build signature."""

    name: str
    left: UserInputConfig
    right: UserInputConfig
    same_build: bool


def representatives() -> tuple[FixtureRow, ...]:
    return tuple(row for row in FIXTURE_ROWS if row.role == "representative")


def rows_for_stack(stack: str) -> tuple[FixtureRow, ...]:
    return tuple(row for row in FIXTURE_ROWS if row.stack == stack)


def read_model_type(model_id: str) -> str:
    """Return the config's model_type without constructing a model."""

    payload = json.loads((_REPO / model_id / "config.json").read_text(encoding="utf-8"))
    model_type = payload.get("model_type")
    if isinstance(model_type, str) and model_type:
        return model_type
    text_config = payload.get("text_config")
    if isinstance(text_config, dict):
        nested = text_config.get("model_type")
        if isinstance(nested, str) and nested:
            return nested
    raise ValueError(f"{model_id} has no model_type")


def baseline_input(row: FixtureRow) -> UserInputConfig:
    """The default one-layer eager input used by the family reuse tests."""

    return UserInputConfig(
        model_id=row.model_id,
        device="TEST_DEVICE",
        num_hidden_layers_override=1,
        do_compile=False,
        quantize_linear_action=QuantizeLinearAction.DISABLED,
        num_queries=2,
        query_len=10,
        context_length=0,
    )


def build_signature(user_input: UserInputConfig) -> L1BuildSignature:
    return L1BuildSignature.from_user_config(user_input)


def build_probes(row: FixtureRow) -> tuple[BuildProbe, ...]:
    """Reuse rules that do not require a forward pass.

    Query length, batch size, and log level stay on one build. Quantization,
    layer override, compile, context length, and device each open a new build.
    """

    base = baseline_input(row)
    return (
        BuildProbe("query_len", base, replace(base, query_len=4), True),
        BuildProbe("num_queries", base, replace(base, num_queries=8), True),
        BuildProbe("log_level", base, replace(base, log_level="DEBUG"), True),
        BuildProbe(
            "quantization",
            base,
            replace(base, quantize_linear_action=QuantizeLinearAction.W8A8_DYNAMIC),
            False,
        ),
        BuildProbe("layers", base, replace(base, num_hidden_layers_override=2), False),
        BuildProbe("compile", base, replace(base, do_compile=True), False),
        BuildProbe("context_length", base, replace(base, context_length=10), False),
        BuildProbe("device", base, replace(base, device="ATLAS_800_A2_376T_64G"), False),
    )


def same_build(left: UserInputConfig, right: UserInputConfig) -> bool:
    return build_signature(left) == build_signature(right)


def representative_matches_family_table() -> bool:
    """The eighteen reuse rows are exactly ``FAMILY_MODEL_IDS``."""

    listed = {row.name: row.model_id for row in representatives()}
    return listed == dict(FAMILY_MODEL_IDS)

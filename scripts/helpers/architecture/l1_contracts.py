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

"""Load the single authoritative L1 contract registry."""

from __future__ import annotations

import json
from pathlib import Path

L1_FAMILIES = frozenset({"qwen", "dense", "moe", "multimodal"})
L1_SECTIONS = frozenset({"scenarios", "assertions", "watchers"})


def load_contract(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    families = payload.get("families")
    if payload.get("schema_version") != 1 or not isinstance(families, dict):
        raise ValueError(f"{path}: invalid L1 contract schema")
    if set(families) != L1_FAMILIES:
        raise ValueError(f"{path}: families must be exactly {sorted(L1_FAMILIES)}")
    return payload


def contract_rows(path: Path, section: str) -> list[dict[str, str]]:
    if section not in L1_SECTIONS:
        raise ValueError(f"unknown L1 contract section: {section}")
    families = load_contract(path)["families"]
    rows = []
    for family, record in families.items():
        entries = record.get(section)
        if not isinstance(entries, list):
            raise ValueError(f"{path}: {family}.{section} must be a list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: {family}.{section} contains a non-object entry")
            rows.append({"family": family, **{key: str(value) for key, value in entry.items()}})
    return rows

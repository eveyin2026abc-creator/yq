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

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.helpers.ci_gate import publish_test_maps as publisher_module
from scripts.helpers.ci_gate.publish_test_maps import (
    Publisher,
    locked_artifacts,
    read_map,
)

_SHA = "a" * 40


def _write_map(path: Path, sha: str = _SHA) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "built_from_commit": sha,
                "map": {"tests/test_demo.py::test_demo": {"package/demo.py": ["run"]}},
            }
        ),
        encoding="utf-8",
    )


def _publisher(tmp_path: Path) -> Publisher:
    return Publisher(
        {
            "source_url": "https://gitcode.com/Ascend/msmodeling.git",
            "publish_url": "publish.git",
            "branches": [],
        },
        tmp_path,
    )


def test_read_map_requires_nonempty_valid_payload(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    _write_map(valid)
    assert read_map(valid)["built_from_commit"] == _SHA

    for index, payload in enumerate(
        (
            {},
            {"schema_version": 1, "built_from_commit": _SHA, "map": {}},
            {
                "schema_version": 1,
                "built_from_commit": "short",
                "map": {"node": {"source": ["symbol"]}},
            },
            {
                "schema_version": 1,
                "built_from_commit": _SHA,
                "map": {"node": {"source": "symbol"}},
            },
        )
    ):
        invalid = tmp_path / f"invalid-{index}.json"
        invalid.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            read_map(invalid)


@pytest.mark.skipif(os.name == "nt", reason="creating symlinks needs elevated Windows privileges")
def test_read_map_rejects_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    link = tmp_path / "test_map.json"
    _write_map(real)
    link.symlink_to(real)

    with pytest.raises(ValueError, match="regular file"):
        read_map(link)


def test_locked_artifacts_rejects_same_version_with_changed_hash() -> None:
    original = b"""version = 1
[[package]]
name = "demo"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
wheels = [{ url = "https://files.example/demo-1.0.whl", hash = "sha256:111", size = 1 }]
"""
    mirrored = original.replace(b"pypi.org", b"mirror.example").replace(b"sha256:111", b"sha256:222")

    assert locked_artifacts(original) != locked_artifacts(mirrored)


def test_locked_artifacts_allows_registry_host_change_with_same_hash() -> None:
    original = b"""version = 1
[[package]]
name = "demo"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
wheels = [{ url = "https://pypi.org/files/demo-1.0.whl", hash = "sha256:111", size = 1 }]
"""
    mirrored = original.replace(b"pypi.org", b"mirror.example")

    assert locked_artifacts(original) == locked_artifacts(mirrored)


def test_generate_keeps_last_good_map_when_sync_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    publisher = _publisher(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    current = tmp_path / "test_map.json"
    _write_map(current)
    calls = 0

    monkeypatch.setattr(publisher_module, "git", lambda *_: "")

    def fail_second_run(*_args: object, **_kwargs: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("sync failed")
        return ""

    monkeypatch.setattr(publisher_module, "run", fail_second_run)

    with pytest.raises(RuntimeError, match="sync failed"):
        publisher.generate(source, "b" * 40, current)

    assert read_map(current)["built_from_commit"] == _SHA


def test_generate_restores_lock_after_dependency_index_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    publisher = _publisher(tmp_path)
    publisher.config["dependency_index"] = "https://example.com/simple"
    source = tmp_path / "source"
    source.mkdir()
    lock = source / "uv.lock"
    original = b"""version = 1
[[package]]
name = "demo"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
wheels = [{ url = "https://pypi.org/files/demo-1.0.whl", hash = "sha256:111", size = 1 }]
"""
    lock.write_bytes(original)
    current = tmp_path / "test_map.json"
    monkeypatch.setattr(publisher_module, "git", lambda *_: "")

    def fake_run(args: list[str], *_args: object, **_kwargs: object) -> str:
        if args[:2] == ["uv", "lock"]:
            lock.write_bytes(original.replace(b"pypi.org", b"example.com"))
        elif args[0] == "bash":
            _write_map(current.with_name("candidate.json"), "b" * 40)
        return ""

    monkeypatch.setattr(publisher_module, "run", fake_run)

    publisher.generate(source, "b" * 40, current)

    assert lock.read_bytes() == original
    assert read_map(current)["built_from_commit"] == "b" * 40


@pytest.mark.parametrize(
    ("mode", "generated", "published"),
    [("generate", True, False), ("publish", False, True), ("all", True, True)],
)
def test_process_separates_generation_from_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    generated: bool,
    published: bool,
) -> None:
    publisher = _publisher(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(publisher, "generate_branch", lambda _branch: calls.append("generate"))
    monkeypatch.setattr(publisher, "publish_branch", lambda _branch: calls.append("publish"))

    publisher.process({"source": "master", "publish": "test-map/master"}, mode)

    assert ("generate" in calls) is generated
    assert ("publish" in calls) is published


def test_branch_discovery_adds_matching_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    publisher = _publisher(tmp_path)
    publisher.config["branches"] = [{"source": "master", "publish": "test-map/master"}]
    publisher.config["discover_branches"] = {
        "patterns": [r"[0-9]+\.[0-9]+\.[0-9]+"],
        "publish_prefix": "test-map/",
    }
    output = "\n".join(
        [
            f"{_SHA}\trefs/heads/26.2.0",
            f"{_SHA}\trefs/heads/feature/skip-me",
            f"{_SHA}\trefs/heads/26.1.0",
        ]
    )
    monkeypatch.setattr(publisher_module, "git", lambda *_: output)

    assert publisher.branches() == [
        {"source": "master", "publish": "test-map/master"},
        {"source": "26.1.0", "publish": "test-map/26.1.0"},
        {"source": "26.2.0", "publish": "test-map/26.2.0"},
    ]


def test_branch_discovery_rejects_publish_target_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    publisher = _publisher(tmp_path)
    publisher.config["branches"] = [{"source": "legacy", "publish": "test-map/26.2.0"}]
    publisher.config["discover_branches"] = {
        "patterns": [r"[0-9]+\.[0-9]+\.[0-9]+"],
        "publish_prefix": "test-map/",
    }
    monkeypatch.setattr(publisher_module, "git", lambda *_: f"{_SHA}\trefs/heads/26.2.0")

    with pytest.raises(ValueError, match="Duplicate effective publish"):
        publisher.branches()


def _write_config(path: Path, payload: dict | None = None) -> Path:
    body = {"source_url": "https://e.com/s.git", "publish_url": "https://e.com/p.git"}
    body["branches"] = [{"source": "master", "publish": "maps/m"}]
    body.update(payload or {})
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"source_url": "-https://example.com/source.git"}, "Missing/invalid source_url"),
        ({"source_url": "https://user:secret@example.com/source.git"}, "Credentials are not allowed"),
        ({"dependency_index": "file:///tmp/simple"}, "dependency_index must be"),
        ({"branches": []}, "at least one branch"),
        ({"branches": [{"source": "master"}]}, "source and publish"),
        ({"branches": [{"source": "m", "publish": "a"}, {"source": "m", "publish": "b"}]}, "Duplicate source"),
        ({"branches": [{"source": "-master", "publish": "maps/master"}]}, "Invalid branch name"),
        ({"branches": [{"source": "bad..name", "publish": "maps/master"}]}, "git failed"),
        ({"discover_branches": {"patterns": ["["]}}, "Invalid branch discovery pattern"),
    ],
)
def test_load_config_rejects_invalid_settings(tmp_path: Path, payload: dict, match: str) -> None:
    with pytest.raises((ValueError, RuntimeError), match=match):
        publisher_module.load_config(_write_config(tmp_path / "publisher.json", payload))

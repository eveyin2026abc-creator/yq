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
import subprocess

import pytest

from scripts.helpers.ci_gate import publish_test_maps as publisher_module
from scripts.helpers.ci_gate.publish_test_maps import Publisher, read_map

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
        {"source_url": "https://gitcode.com/Ascend/msmodeling.git", "publish_url": "publish.git", "branches": []},
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
            {"schema_version": 1, "built_from_commit": "short", "map": {"node": {"source": ["symbol"]}}},
            {"schema_version": 1, "built_from_commit": _SHA, "map": {"node": {"source": "symbol"}}},
        )
    ):
        invalid = tmp_path / f"invalid-{index}.json"
        invalid.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            read_map(invalid)


def test_process_reuses_published_map_when_source_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    publisher = _publisher(tmp_path)
    branch = {"source": "master", "publish": "maps/master"}
    root = tmp_path / publisher_module.hashlib.sha256(b"master").hexdigest()[:16]

    def refresh(repo: Path, _branch: str) -> None:
        _write_map(repo / "test_map.json")

    monkeypatch.setattr(publisher, "refresh_artifacts", refresh)
    monkeypatch.setattr(publisher, "repository", lambda *_: None)
    monkeypatch.setattr(publisher_module, "git", lambda _root, *args: _SHA if args[0] == "rev-parse" else "")
    monkeypatch.setattr(publisher, "generate", lambda *_: pytest.fail("unchanged map must not regenerate"))
    monkeypatch.setattr(publisher, "publish", lambda *_: pytest.fail("unchanged map must not republish"))

    publisher.process(branch)

    assert read_map(root / "test_map.json")["built_from_commit"] == _SHA


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


def test_cycle_continues_after_one_branch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    publisher = _publisher(tmp_path)
    publisher.config["branches"] = [
        {"source": "broken", "publish": "maps/broken"},
        {"source": "master", "publish": "maps/master"},
    ]
    visited: list[str] = []

    def process(branch: dict) -> None:
        visited.append(branch["source"])
        if branch["source"] == "broken":
            raise RuntimeError("expected")

    monkeypatch.setattr(publisher, "process", process)

    assert publisher.cycle() is False
    assert visited == ["broken", "master"]


@pytest.mark.skipif(os.name == "nt", reason="publisher is a Linux service")
def test_publisher_end_to_end_updates_only_after_source_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_work = tmp_path / "source-work"
    source_bare = tmp_path / "source.git"
    publish_bare = tmp_path / "publish.git"
    source_work.mkdir()
    subprocess.run(["git", "init", "-b", "master"], cwd=source_work, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source_work, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source_work, check=True)
    (source_work / "source.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.txt"], cwd=source_work, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=source_work, check=True, capture_output=True)
    subprocess.run(["git", "init", "--bare", source_bare], check=True, capture_output=True)
    subprocess.run(["git", "remote", "add", "origin", source_bare], cwd=source_work, check=True)
    subprocess.run(["git", "push", "origin", "master"], cwd=source_work, check=True, capture_output=True)
    subprocess.run(["git", "init", "--bare", publish_bare], check=True, capture_output=True)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "uv").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (fake_bin / "bash").write_text(
        """#!/bin/sh
sha=$(git rev-parse publisher-target)
printf '{"schema_version":1,"built_from_commit":"%s","map":{"tests/test_demo.py::test_demo":{"package/demo.py":["run"]}}}\n' "$sha" > "$MSMODELING_TEST_MAP_PATH"
""",
        encoding="utf-8",
    )
    (fake_bin / "uv").chmod(0o755)
    (fake_bin / "bash").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.com")

    publisher = Publisher(
        {
            "source_url": str(source_bare),
            "publish_url": str(publish_bare),
            "branches": [{"source": "master", "publish": "test-map/master"}],
        },
        tmp_path / "publisher",
    )
    assert publisher.cycle() is True
    first_publish = subprocess.run(
        ["git", "--git-dir", publish_bare, "rev-parse", "refs/heads/test-map/master"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert publisher.cycle() is True
    assert (
        subprocess.run(
            ["git", "--git-dir", publish_bare, "rev-parse", "refs/heads/test-map/master"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == first_publish
    )

    (source_work / "source.txt").write_text("two\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "update"], cwd=source_work, check=True, capture_output=True)
    subprocess.run(["git", "push", "origin", "master"], cwd=source_work, check=True, capture_output=True)
    assert publisher.cycle() is True
    published = subprocess.run(
        ["git", "--git-dir", publish_bare, "show", "refs/heads/test-map/master:test_map.json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_work, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert json.loads(published)["built_from_commit"] == source_sha

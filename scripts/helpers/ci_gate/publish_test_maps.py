#!/usr/bin/env python3

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
"""Serial Linux worker publishing one test_map per configured source branch."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import tempfile
import time
import tomllib
from typing import Iterator
from urllib.parse import urlsplit

LOG = logging.getLogger("test_map_publisher")


def run(
    args: list[str],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
    timeout: int = 120,
    capture: bool = True,
) -> str:
    """Run without a shell and terminate the entire child group on interruption."""
    with subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        text=True,
        start_new_session=True,
    ) as proc:
        try:
            output, _ = proc.communicate(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
        if proc.returncode:
            raise RuntimeError(f"{args[0]} failed (exit {proc.returncode})")
        return output.strip() if output is not None else ""


def git(root: Path, *args: str) -> str:
    return run(["git", *args], root)


def read_regular_bytes(path: Path) -> bytes:
    """Read a regular file without following a final-component symlink."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        raise
    if not stat.S_ISREG(mode):
        raise ValueError(f"Expected a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"Expected a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def atomic_copy_regular(source: Path, target: Path) -> None:
    """Copy a regular file via a same-directory temporary file."""
    payload = read_regular_bytes(source)
    if target.exists() or target.is_symlink():
        if not stat.S_ISREG(target.lstat().st_mode):
            raise ValueError(f"Refusing to replace a non-regular file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o640)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_map(path: Path) -> dict:
    """Reject incomplete/corrupt maps before using or publishing them."""
    payload = json.loads(read_regular_bytes(path).decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported test_map schema")
    sha = payload.get("built_from_commit", "")
    if not isinstance(sha, str) or len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Invalid built_from_commit")
    mapping = payload.get("map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Empty or invalid test_map")
    for node, sources in mapping.items():
        if not isinstance(node, str) or not isinstance(sources, dict) or not sources:
            raise ValueError("Invalid test_map node")
        for source, symbols in sources.items():
            if not isinstance(source, str) or not isinstance(symbols, list) or any(type(s) is not str for s in symbols):
                raise ValueError("Invalid test_map source symbols")
    return payload


def locked_artifacts(data: bytes) -> list[tuple]:
    """Return immutable package identities while allowing only registry host changes."""
    packages = tomllib.loads(data.decode("utf-8")).get("package", [])
    identities = []
    for package in packages:
        source = package.get("source", {})
        source_identity = tuple(sorted(source.items())) if "registry" not in source else (("registry", "<mirror>"),)
        artifacts = []
        for artifact in [package.get("sdist"), *package.get("wheels", [])]:
            if not artifact:
                continue
            digest = artifact.get("hash")
            url = artifact.get("url", "")
            if not isinstance(digest, str) or not digest.startswith("sha256:"):
                raise ValueError(f"Locked artifact lacks a sha256 hash: {package.get('name', '<unknown>')}")
            filename = Path(urlsplit(url).path).name
            if not filename:
                raise ValueError(f"Locked artifact lacks a filename: {package.get('name', '<unknown>')}")
            artifacts.append((filename, digest, artifact.get("size")))
        if "registry" in source and not artifacts:
            raise ValueError(f"Registry package lacks locked artifacts: {package.get('name', '<unknown>')}")
        identities.append(
            (
                package["name"],
                package.get("version", ""),
                source_identity,
                tuple(sorted(artifacts)),
            )
        )
    return sorted(identities)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("source_url", "publish_url"):
        if not isinstance(config.get(key), str) or not config[key] or config[key].startswith("-"):
            raise ValueError(f"Missing/invalid {key}")
        parsed = urlsplit(config[key])
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(f"Credentials are not allowed in {key}; use a Git credential helper")
    dependency_index = config.get("dependency_index")
    if dependency_index is not None:
        parsed = urlsplit(dependency_index)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username is not None:
            raise ValueError("dependency_index must be an HTTP(S) URL without credentials")
    branches = config.get("branches")
    if not isinstance(branches, list) or not branches:
        raise ValueError("Configure at least one branch")
    if not all(isinstance(branch, dict) and {"source", "publish"} <= branch.keys() for branch in branches):
        raise ValueError("Each branch needs source and publish names")
    for field in ("source", "publish"):
        names = [branch[field] for branch in branches]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate {field} branches")
        for name in names:
            if not isinstance(name, str) or name.startswith("-"):
                raise ValueError("Invalid branch name")
            git(path.parent, "check-ref-format", f"refs/heads/{name}")
    discovery = config.get("discover_branches")
    if discovery is not None:
        if not isinstance(discovery, dict) or not isinstance(discovery.get("patterns"), list):
            raise ValueError("discover_branches needs a patterns list")
        patterns = discovery["patterns"]
        if not patterns or not all(isinstance(pattern, str) and pattern for pattern in patterns):
            raise ValueError("Branch discovery patterns must be non-empty strings")
        try:
            for pattern in patterns:
                re.compile(pattern)
        except re.error as error:
            raise ValueError(f"Invalid branch discovery pattern: {error}") from error
        prefix = discovery.get("publish_prefix", "test-map/")
        if not isinstance(prefix, str) or not prefix or prefix.startswith("-"):
            raise ValueError("Invalid discovery publish_prefix")
        git(path.parent, "check-ref-format", f"refs/heads/{prefix}placeholder")
    return config


@contextmanager
def worker_lock(workspace: Path, mode: str = "all") -> Iterator[None]:
    """One worker owns every branch directory; releases automatically after crashes."""
    import fcntl  # Linux server only.

    lock_root = workspace if mode == "all" else workspace / ("generation" if mode == "generate" else "publication")
    lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / "publisher.lock").open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Publisher:
    """Keep isolated source/environment/artifact checkouts and a durable local outbox."""

    def __init__(
        self,
        config: dict,
        workspace: Path,
        *,
        python: str = "3.11",
        timeout: int = 7200,
    ):
        self.config = config
        self.workspace = workspace
        self.python = python
        self.timeout = timeout

    def repository(self, path: Path, url: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if not (path / ".git").exists():
            git(path, "init")
            git(path, "remote", "add", "origin", url)
        elif git(path, "remote", "get-url", "origin") != url:
            raise ValueError("Repository URL changed; use a new workspace")

    def branch_paths(self, name: str) -> tuple[Path, Path, Path]:
        key = hashlib.sha256(name.encode()).hexdigest()[:16]
        generation = self.workspace / "generation" / key
        source = generation / "source"
        current = generation / "test_map.json"
        artifacts = self.workspace / "publication" / key / "artifacts"
        return source, current, artifacts

    def refresh_artifacts(self, repo: Path, branch: str) -> None:
        self.repository(repo, self.config["publish_url"])
        ref = f"refs/heads/{branch}"
        if git(repo, "ls-remote", "--heads", "origin", ref):
            git(repo, "fetch", "origin", f"+{ref}:refs/remotes/origin/published")
            git(repo, "checkout", "-B", "published", "refs/remotes/origin/published")
        elif not git(repo, "branch", "--list", "published"):
            git(repo, "checkout", "--orphan", "published")

    def publish(self, repo: Path, branch: str, test_map_path: Path) -> None:
        payload = read_map(test_map_path)
        target = repo / "test_map.json"
        test_map_bytes = read_regular_bytes(test_map_path)
        if not target.exists() or read_regular_bytes(target) != test_map_bytes:
            atomic_copy_regular(test_map_path, target)
            git(repo, "add", "--", "test_map.json")
        if git(repo, "status", "--porcelain", "--", "test_map.json"):
            git(
                repo,
                "commit",
                "-s",
                "-m",
                f"data: update test_map for {payload['built_from_commit']}",
            )
        git(repo, "push", "origin", f"HEAD:refs/heads/{branch}")
        LOG.info("Published %s at source %s", branch, payload["built_from_commit"])

    def generate(self, source: Path, sha: str, current: Path) -> None:
        git(source, "checkout", "--detach", sha)
        git(source, "branch", "-f", "publisher-target", sha)
        remotes = git(source, "remote").splitlines()
        if "snapshot" not in remotes:
            git(source, "remote", "add", "snapshot", str(source / ".git"))
        candidate = current.with_name("candidate.json")
        candidate.unlink(missing_ok=True)
        if current.exists():
            atomic_copy_regular(current, candidate)
        env = dict(os.environ)
        for key in (
            "PYTHON",
            "UV_NO_SYNC",
            "UV_PROJECT_ENVIRONMENT",
            "PYTHONPATH",
            "MSMODELING_TEST_MAP_TARGET_BRANCH",
            "BASH_ENV",
            "ENV",
        ):
            env.pop(key, None)
        env.update(
            MSMODELING_TEST_MAP_PATH=str(candidate),
            MSMODELING_TEST_BASE_BRANCH="snapshot/publisher-target",
            MSMODELING_TEST_MAP_TARGET_BRANCH="snapshot/publisher-target",
            GIT_TERMINAL_PROMPT="0",
        )
        LOG.info("Synchronizing dependencies for source %s", sha)
        lock = source / "uv.lock"
        dependency_index = self.config.get("dependency_index")
        original_lock = lock.read_bytes() if dependency_index else None
        try:
            if dependency_index:
                assert original_lock is not None
                run(
                    ["uv", "lock", "--default-index", dependency_index],
                    source,
                    env=env,
                    timeout=self.timeout,
                    capture=False,
                )
                if locked_artifacts(lock.read_bytes()) != locked_artifacts(original_lock):
                    raise ValueError("Dependency index changed locked package artifacts")
            run(
                ["uv", "sync", "--python", self.python, "--frozen", "--group", "ci"],
                source,
                env=env,
                timeout=self.timeout,
                capture=False,
            )
        finally:
            if original_lock is not None:
                lock.write_bytes(original_lock)
        env.update(PYTHON=str(source / ".venv/bin/python"), UV_NO_SYNC="1")
        run(
            [
                "bash",
                "scripts/run_test_map_sync.sh",
                "--once",
                "--target-branch",
                "snapshot/publisher-target",
            ],
            source,
            env=env,
            timeout=self.timeout,
            capture=False,
        )
        if read_map(candidate)["built_from_commit"] != sha:
            raise ValueError("Generated map does not match pinned source commit")
        candidate.replace(current)

    def generate_branch(self, branch: dict) -> None:
        name = branch["source"]
        source, current, _artifacts = self.branch_paths(name)
        self.repository(source, self.config["source_url"])
        git(source, "fetch", "origin", f"+refs/heads/{name}:refs/remotes/origin/target")
        sha = git(source, "rev-parse", "refs/remotes/origin/target^{commit}")
        if current.exists() and read_map(current)["built_from_commit"] == sha:
            LOG.info("Unchanged: %s at %s", name, sha)
            return
        LOG.info("Generating %s at %s", name, sha)
        self.generate(source, sha, current)

    def publish_branch(self, branch: dict) -> None:
        _source, current, artifacts = self.branch_paths(branch["source"])
        if not current.exists():
            LOG.info("No generated map ready for %s", branch["source"])
            return
        read_map(current)
        self.refresh_artifacts(artifacts, branch["publish"])
        remote_map = artifacts / "test_map.json"
        if remote_map.exists() and read_regular_bytes(current) == read_regular_bytes(remote_map):
            LOG.info("Already published: %s", branch["publish"])
            return
        self.publish(artifacts, branch["publish"], current)

    def process(self, branch: dict, mode: str = "all") -> None:
        if mode in {"all", "generate"}:
            self.generate_branch(branch)
        if mode in {"all", "publish"}:
            self.publish_branch(branch)

    def branches(self) -> list[dict[str, str]]:
        branches = list(self.config["branches"])
        discovery = self.config.get("discover_branches")
        if discovery is None:
            return branches
        output = git(self.workspace, "ls-remote", "--heads", self.config["source_url"])
        patterns = [re.compile(pattern) for pattern in discovery["patterns"]]
        prefix = discovery.get("publish_prefix", "test-map/")
        configured = {branch["source"] for branch in branches}
        discovered = []
        for line in output.splitlines():
            if "refs/heads/" not in line:
                continue
            name = line.split("refs/heads/", 1)[1]
            if name in configured:
                continue
            if any(pattern.fullmatch(name) for pattern in patterns):
                discovered.append(name)
        discovered.sort()
        branches.extend({"source": name, "publish": f"{prefix}{name}"} for name in discovered)
        for field in ("source", "publish"):
            names = [branch[field] for branch in branches]
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate effective {field} branches")
        LOG.info(
            "Active source branches: %s",
            ", ".join(branch["source"] for branch in branches),
        )
        return branches

    def cycle(self, mode: str = "all") -> bool:
        success = True
        try:
            branches = self.branches()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            LOG.exception("Branch discovery failed; retrying next cycle")
            return False
        for branch in branches:
            try:
                self.process(branch, mode)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                LOG.exception("Branch failed: %s; retrying next cycle", branch["source"])
                success = False
        return success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--python", default="3.11")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--mode", choices=("all", "generate", "publish"), default="all")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0 or args.timeout <= 0:
        parser.error("interval and timeout must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")

    def stop(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        config = load_config(args.config.resolve())
        with worker_lock(args.workspace.resolve(), args.mode):
            publisher = Publisher(
                config,
                args.workspace.resolve(),
                python=args.python,
                timeout=args.timeout,
            )
            while True:
                success = publisher.cycle(args.mode)
                if args.once:
                    return 0 if success else 1
                time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError):
        LOG.exception("Publisher stopped")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

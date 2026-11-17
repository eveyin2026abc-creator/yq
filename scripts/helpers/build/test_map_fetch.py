"""Resolve and download the external test_map used by CI gate.

The default URL is the published map on ``eveyin1/msmodeling_pr``, branch
``test-map/{base_branch}``. Same source as master.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Final
from urllib.parse import quote, urlsplit

from scripts.helpers._paths import REPO_ROOT

logger: Final = logging.getLogger("build.test_map")

_DOWNLOAD_TIMEOUT_SECONDS: Final = 30
_TEST_MAP_FILENAME: Final = "test_map.json"
_DEFAULT_CACHE: Final = ".msmodeling_cache"
_TEST_MAP_URL_TEMPLATE: Final = (
    "https://gitcode.com/api/v5/repos/eveyin1/msmodeling_pr/raw/test_map.json?ref=test-map/{branch}"
)
_TEST_MAP_ALLOWED_HOSTS: Final = frozenset({"gitcode.com"})


class MapFetchError(Exception):
    """Raised when test_map cannot be resolved or downloaded."""


def build_test_map_url(base_branch: str) -> str:
    """Return the eveyin1 test_map URL for *base_branch*."""
    branch = base_branch.strip().lstrip("/")
    if not branch:
        raise ValueError("base_branch must be non-empty")
    template = os.environ.get("MSMODELING_TEST_MAP_URL_TEMPLATE", _TEST_MAP_URL_TEMPLATE).strip()
    _validate_test_map_template(template)
    url = template.format(branch=quote(branch, safe="/"))
    hostname = urlsplit(url).hostname
    if hostname is None or hostname.lower() not in _allowed_test_map_hosts():
        raise ValueError(f"test_map URL host {hostname!r} is not allowed")
    return url


def default_test_map_path(*, cache_dir: str | Path | None = None, base_branch: str) -> Path:
    """Return ``{cache}/test_map/{base_branch}/test_map.json``."""
    root = Path(cache_dir) if cache_dir is not None else Path(os.environ.get("MSMODELING_CACHE", _DEFAULT_CACHE))
    branch = base_branch.strip().lstrip("/")
    if not branch:
        raise ValueError("base_branch must be non-empty")
    return root / "test_map" / branch / _TEST_MAP_FILENAME


def resolve_test_map_path(
    *,
    configured: str | None,
    base_branch: str,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return a usable test_map file, downloading from eveyin1 when needed.

    A configured path is used only when it is an existing file with a JSON
    object root. Otherwise the branch-scoped cache file is used, and missing
    or corrupt cache is replaced by a download.
    """
    raw = configured.strip() if configured else ""
    if raw:
        path = Path(raw)
        if path.is_file():
            try:
                _validate_test_map_bytes(path.read_bytes(), source=str(path))
            except MapFetchError as exc:
                logger.warning("%s; re-downloading for base branch %r.", exc, base_branch)
            else:
                logger.info("Using test_map at %s", path)
                return path
        else:
            logger.warning(
                "MSMODELING_TEST_MAP_PATH=%s is not a readable file; downloading test_map for base branch %r instead.",
                raw,
                base_branch,
            )
    else:
        logger.warning(
            "MSMODELING_TEST_MAP_PATH is unset; downloading test_map for base branch %r.",
            base_branch,
        )

    dest = default_test_map_path(cache_dir=cache_dir, base_branch=base_branch)
    if not dest.is_absolute():
        dest = REPO_ROOT / dest

    if dest.is_file():
        try:
            _validate_test_map_bytes(dest.read_bytes(), source=str(dest))
        except MapFetchError as exc:
            logger.warning("%s; re-downloading.", exc)
        else:
            logger.info("Using cached test_map at %s", dest)
            return dest

    download_test_map(base_branch=base_branch, dest=dest)
    return dest


def download_test_map(*, base_branch: str, dest: Path) -> None:
    """Download the eveyin1 test_map for *base_branch* to *dest*."""
    try:
        url = build_test_map_url(base_branch)
    except ValueError as exc:
        raise MapFetchError(f"invalid test_map download URL for branch {base_branch!r}: {exc}") from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    logger.info("Downloading test_map from %s → %s", url, dest)
    try:
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            body = response.read()
        _validate_test_map_bytes(body, source=url)
        tmp.write_bytes(body)
        tmp.replace(dest)
    except MapFetchError:
        _cleanup_tmp(tmp)
        raise
    except urllib.error.HTTPError as exc:
        _cleanup_tmp(tmp)
        raise MapFetchError(f"failed to download test_map from {url}: HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        _cleanup_tmp(tmp)
        raise MapFetchError(f"failed to download test_map from {url}: {exc.reason}") from exc
    except OSError as exc:
        _cleanup_tmp(tmp)
        raise MapFetchError(f"failed to download test_map from {url} or write to {dest}: {exc}") from exc
    logger.info("test_map ready at %s (%d bytes)", dest, dest.stat().st_size)


def _validate_test_map_template(template: str) -> None:
    if template.count("{branch}") != 1:
        raise ValueError("MSMODELING_TEST_MAP_URL_TEMPLATE must contain one {branch} placeholder")
    try:
        url = template.format(branch="branch")
    except (KeyError, ValueError) as exc:
        raise ValueError("MSMODELING_TEST_MAP_URL_TEMPLATE contains an invalid placeholder") from exc
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("MSMODELING_TEST_MAP_URL_TEMPLATE must produce an HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Credentials are not allowed in MSMODELING_TEST_MAP_URL_TEMPLATE")


def _allowed_test_map_hosts() -> frozenset[str]:
    raw = os.environ.get("MSMODELING_TEST_MAP_ALLOWED_HOSTS")
    if raw is None:
        return _TEST_MAP_ALLOWED_HOSTS
    hosts = frozenset(host.strip().lower() for host in raw.split(",") if host.strip())
    if not hosts:
        raise ValueError("MSMODELING_TEST_MAP_ALLOWED_HOSTS must contain at least one hostname")
    return hosts


def _validate_test_map_bytes(body: bytes, *, source: str) -> None:
    if not body:
        raise MapFetchError(f"test_map is empty: {source}")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MapFetchError(f"test_map is not valid UTF-8 JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise MapFetchError(f"test_map JSON root must be an object: {source}")


def _cleanup_tmp(tmp: Path) -> None:
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        return

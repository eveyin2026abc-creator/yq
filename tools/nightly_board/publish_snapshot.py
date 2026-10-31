#!/usr/bin/env python3
"""Export and optionally commit/push a nightly-board static snapshot."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import export_static
import server


ROOT = Path(__file__).resolve().parent


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT.parent.parent,
        check=check,
        text=True,
        capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="commit and push the generated public snapshot")
    parser.add_argument("--remote", default="fork")
    parser.add_argument("--branch", default="feature/nightly-board-static")
    args = parser.parse_args()

    output = ROOT / "public"
    count = export_static.export(output)
    print(f"exported {count} run(s) to {output}")
    if not args.publish:
        return

    relative = output.relative_to(ROOT.parent.parent)
    git("add", str(relative))
    changed = git("diff", "--cached", "--quiet", check=False).returncode != 0
    if not changed:
        print("snapshot unchanged; nothing to publish")
        return

    date = datetime.now(server.BEIJING).date().isoformat()
    git("commit", "-m", f"data: update nightly board snapshot {date}")
    git("push", args.remote, f"HEAD:refs/heads/{args.branch}")
    print(f"published snapshot to {args.remote}/{args.branch}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export local nightly logs and issue states as a privacy-conscious static site."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import server


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def export(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(server.ROOT / "index.html", output / "index.html")

    runs = server.load_runs()
    dates = sorted({run["date"] for run in runs})
    months = sorted({date[:7] for date in dates} | {server.beijing_today()[:7]})

    for month in months:
        year, month_number = (int(item) for item in month.split("-"))
        write_json(output / "data" / "month" / f"{month}.json", server.month_payload(year, month_number))

    for date in dates:
        write_json(output / "data" / "day" / f"{date}.json", server.day_payload(date))

    exported_at = datetime.now(server.BEIJING).isoformat(timespec="seconds")
    config = (
        "window.NIGHTLY_BOARD_STATIC = true;\n"
        f"window.NIGHTLY_BOARD_TODAY = {json.dumps(server.beijing_today())};\n"
        f"window.NIGHTLY_BOARD_EXPORTED_AT = {json.dumps(exported_at)};\n"
    )
    (output / "config.js").write_text(config, encoding="utf-8")
    write_json(
        output / "data" / "manifest.json",
        {"exported_at": exported_at, "dates": dates, "months": months, "run_count": len(runs)},
    )
    return len(runs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "public")
    args = parser.parse_args()
    count = export(args.output.resolve())
    print(f"exported {count} run(s) to {args.output.resolve()}")


if __name__ == "__main__":
    main()

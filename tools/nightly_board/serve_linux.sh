#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PORT=${NIGHTLY_BOARD_PORT:-8766}
HOST=${NIGHTLY_BOARD_HOST:-127.0.0.1}
exec python3 -m http.server "$PORT" --bind "$HOST" --directory "$ROOT/public"

#!/usr/bin/env bash
# Run the local Pokemon Showdown server with no auth/rate-limiting, for
# offline self-play and evaluation. Listens on ws://localhost:8000.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_DIR="$ROOT_DIR/third_party/pokemon-showdown"
PORT="${1:-8000}"

if [ ! -d "$SERVER_DIR" ]; then
  echo "[run_server] $SERVER_DIR not found. Run: bash scripts/setup_server.sh" >&2
  exit 1
fi

echo "[run_server] Starting Showdown on port $PORT (--no-security) ..."
cd "$SERVER_DIR"
exec node pokemon-showdown start --no-security --port "$PORT"

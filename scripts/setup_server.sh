#!/usr/bin/env bash
# Clone and install a local Pokemon Showdown server for offline play/training.
# This is poke-env's standard local target. Idempotent: safe to re-run.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$ROOT_DIR/third_party"
SERVER_DIR="$VENDOR_DIR/pokemon-showdown"
REPO_URL="https://github.com/smogon/pokemon-showdown.git"

mkdir -p "$VENDOR_DIR"

if [ ! -d "$SERVER_DIR/.git" ]; then
  echo "[setup_server] Cloning pokemon-showdown into $SERVER_DIR ..."
  git clone --depth 1 "$REPO_URL" "$SERVER_DIR"
else
  echo "[setup_server] pokemon-showdown already present; pulling latest ..."
  git -C "$SERVER_DIR" pull --ff-only || true
fi

echo "[setup_server] Installing npm dependencies (this can take a few minutes) ..."
( cd "$SERVER_DIR" && npm install )

echo "[setup_server] Done. Start the server with: bash scripts/run_server.sh"

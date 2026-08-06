#!/usr/bin/env bash
# Clone and install a local Pokemon Showdown server for offline play/training.
# This is poke-env's standard local target. Idempotent: safe to re-run.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$ROOT_DIR/third_party"
SERVER_DIR="$VENDOR_DIR/pokemon-showdown"
REPO_URL="https://github.com/smogon/pokemon-showdown.git"

# PINNED, deliberately. third_party/ is gitignored, so the simulator is NOT part of a clone and
# used to be fetched at upstream HEAD -- which silently re-dated the whole re-simulation stack.
# gen9randombattle SETS CHANGE UPSTREAM, so a fixed PRNG seed generates different teams under a
# different rev, and every recorded inputlog (the R-PREDICT fidelity + resim test fixtures, and
# any cached replay) goes illegal partway through. That is exactly how the drift between the
# 2026-06-28 Gate A pass and a 2026-07-19 working copy broke both env-gated tests.
#
# Bumping this pin is a DELIBERATE act: it invalidates tests/conftest.py's inputlog fixture, which
# must be regenerated against the new rev in the same commit. Full SHA, not abbreviated -- the
# fetch below asks the server for this object by name and the protocol needs the full one.
SHOWDOWN_REV="${SHOWDOWN_REV:-393d5c8675f28cacceea1fb1a4d976205c3540b1}"

mkdir -p "$VENDOR_DIR"

if [ ! -d "$SERVER_DIR/.git" ]; then
  echo "[setup_server] Initialising $SERVER_DIR ..."
  git init -q "$SERVER_DIR"
  git -C "$SERVER_DIR" remote add origin "$REPO_URL"
fi

# Fetch ONLY the pinned commit (GitHub serves arbitrary SHAs), so this stays as cheap as the
# `--depth 1` clone it replaces. Skipped entirely when the object is already local, which is the
# common case: an existing working copy already sitting on the pin does no network I/O at all.
if ! git -C "$SERVER_DIR" cat-file -e "${SHOWDOWN_REV}^{commit}" 2>/dev/null; then
  echo "[setup_server] Fetching pinned rev $SHOWDOWN_REV ..."
  git -C "$SERVER_DIR" fetch --depth 1 origin "$SHOWDOWN_REV"
fi
echo "[setup_server] Checking out pinned rev $SHOWDOWN_REV ..."
git -C "$SERVER_DIR" checkout --detach "$SHOWDOWN_REV"

echo "[setup_server] Installing npm dependencies (this can take a few minutes) ..."
( cd "$SERVER_DIR" && npm install )

echo "[setup_server] Done. Start the server with: bash scripts/run_server.sh"

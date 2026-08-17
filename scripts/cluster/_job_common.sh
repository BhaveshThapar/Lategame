#!/usr/bin/env bash
# Shared job prologue: bring up a PRIVATE Showdown for this job and tear it down on exit.
#
# poke-env pins ws://localhost:8000, so two jobs on one host would share a server and silently
# battle into each other's games. Each job therefore gets its own port, and rotomai/config.py
# reads it from ROTOMAI_SHOWDOWN_PORT.
#
# Sourced, not executed:  source scripts/cluster/_job_common.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"

# sbatch does NOT inherit an interactive shell's `conda activate` -- without this every job
# silently runs against whatever bare `python` happens to be on PATH (wrong deps, or none).
# Mirrors setup_umiacs.sh's own conda discovery: PATH first, then the private bootstrap it
# creates at $REPO_DIR/.miniforge3 when the cluster has no conda module at all (measured on
# UMIACS Nexus: no module, no system install).
activate_env() {
  local base
  if command -v conda >/dev/null 2>&1; then
    base="$(conda info --base)"
  elif [[ -x "$REPO_DIR/.miniforge3/bin/conda" ]]; then
    base="$REPO_DIR/.miniforge3"
  else
    echo "[job] FATAL: no conda found on PATH or at $REPO_DIR/.miniforge3." >&2
    echo "[job]        Run scripts/cluster/setup_umiacs.sh first." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$base/etc/profile.d/conda.sh"
  conda activate rotomai
}
activate_env
echo "[job] python: $(command -v python)  ($(python --version 2>&1))"

# Port from the array index, so tasks on the same node cannot collide. 8000 stays the
# interactive default; jobs live above it.
#
# The array index alone is only unique WITHIN one array. Build 23 runs two arms as two concurrent
# 0-2 arrays: without a distinct base they both compute 8100-8102, and any two tasks landing on the
# same node share a server -- poke-env does not error, it silently battles into the other arm's
# games, contaminating the exact comparison the build exists to make. So give each concurrent array
# its own base:  ROTOMAI_SHOWDOWN_PORT_BASE=8200 sbatch --array=0-2 ...
# Set the BASE, never ROTOMAI_SHOWDOWN_PORT itself -- that would pin all three tasks to one port.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
export ROTOMAI_SHOWDOWN_PORT="${ROTOMAI_SHOWDOWN_PORT:-$((${ROTOMAI_SHOWDOWN_PORT_BASE:-8100} + TASK_ID))}"

# Server logs live under logs/showdown/<bucket>/, not the repo root. Resolved HERE rather than at
# source time: callers set BUILD *after* sourcing this file (ppo_seed.slurm does), so a source-time
# default would always miss it and bucket every arm under the job name instead. LOG_BUCKET wins if
# set; then BUILD; then the job name with sbatch's `ra-` prefix stripped, which is what the
# non-ppo jobs (strength, b22-stageA, probe-rep) sort by.
_resolve_server_log() {
  local bucket="${LOG_BUCKET:-${BUILD:-${SLURM_JOB_NAME:-local}}}"
  SERVER_LOG="${SERVER_LOG:-logs/showdown/${bucket#ra-}/showdown_${SLURM_JOB_ID:-local}_${TASK_ID}.log}"
  mkdir -p "$(dirname "$SERVER_LOG")"
}

start_showdown() {
  _resolve_server_log
  echo "[job] starting Showdown on port ${ROTOMAI_SHOWDOWN_PORT}"
  bash scripts/run_server.sh "$ROTOMAI_SHOWDOWN_PORT" > "$SERVER_LOG" 2>&1 &
  SHOWDOWN_PID=$!
  # Do NOT proceed until the port actually accepts connections. poke-env does not fail fast on a
  # dead server -- it HANGS, for hours, which is how a whole run gets silently wasted.
  for _ in $(seq 1 60); do
    if python -c "
import socket,sys
s=socket.socket()
s.settimeout(1)
sys.exit(0 if s.connect_ex(('127.0.0.1', ${ROTOMAI_SHOWDOWN_PORT})) == 0 else 1)
" 2>/dev/null; then
      echo "[job] Showdown up (pid $SHOWDOWN_PID)"
      return 0
    fi
    if ! kill -0 "$SHOWDOWN_PID" 2>/dev/null; then
      echo "[job] FATAL: Showdown died on startup. Log:" >&2
      tail -20 "$SERVER_LOG" >&2
      exit 1
    fi
    sleep 2
  done
  echo "[job] FATAL: Showdown never opened port ${ROTOMAI_SHOWDOWN_PORT} after 120s" >&2
  tail -20 "$SERVER_LOG" >&2
  exit 1
}

stop_showdown() {
  if [[ -n "${SHOWDOWN_PID:-}" ]] && kill -0 "$SHOWDOWN_PID" 2>/dev/null; then
    echo "[job] stopping Showdown (pid $SHOWDOWN_PID)"
    kill "$SHOWDOWN_PID" 2>/dev/null || true
    wait "$SHOWDOWN_PID" 2>/dev/null || true
  fi
}
# Fires on success, failure, and scancel alike -- a leaked node server would poison the next job
# that lands on this host.
trap stop_showdown EXIT

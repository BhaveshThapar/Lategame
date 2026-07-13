#!/usr/bin/env bash
# One-time cluster setup: conda env + a built Showdown.
#
# UNVERIFIED ON UMIACS -- the module names below are a guess and are the one part of this that was
# not measured. Everything else (the port override, the server handshake, the job scripts) is tested
# locally. Adjust the module lines to whatever `module avail` actually reports, then run once.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

echo "== node =="
# Showdown needs node. Prefer a cluster module; fall back to conda's nodejs if there is none.
if command -v module >/dev/null 2>&1; then
  module load nodejs 2>/dev/null || module load node 2>/dev/null || \
    echo "  no nodejs module found -- will rely on conda's"
fi
if ! command -v node >/dev/null 2>&1; then
  echo "  installing nodejs into the conda env"
  conda install -y -c conda-forge nodejs
fi
node --version

echo "== conda env =="
if conda env list | grep -q '^lategame '; then
  echo "  'lategame' exists; updating from environment.yml"
  conda env update -n lategame -f environment.yml --prune
else
  conda env create -f environment.yml
fi

echo "== showdown =="
if [ ! -d third_party/pokemon-showdown ]; then
  bash scripts/setup_server.sh
else
  echo "  third_party/pokemon-showdown present; rebuilding"
  (cd third_party/pokemon-showdown && npm run build)
fi

echo "== smoke test: does a server come up on a NON-default port? =="
# This is the whole point of the port override -- if it fails here, the sbatch array will silently
# have every task fighting over :8000.
export LATEGAME_SHOWDOWN_PORT=8199
bash scripts/run_server.sh "$LATEGAME_SHOWDOWN_PORT" > /tmp/lg_setup_showdown.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  if python -c "
import socket,sys
s=socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(('127.0.0.1', $LATEGAME_SHOWDOWN_PORT)) == 0 else 1)" 2>/dev/null; then
    echo "  OK -- Showdown answered on :$LATEGAME_SHOWDOWN_PORT"
    python -c "
from lategame.config import LOCAL_SERVER
assert ':8199/' in LOCAL_SERVER.websocket_url, LOCAL_SERVER
print('  OK -- lategame.config points at', LOCAL_SERVER.websocket_url)"
    echo
    echo "setup complete. next:  sbatch scripts/cluster/stage_a.slurm"
    exit 0
  fi
  sleep 2
done
echo "FAILED: Showdown never opened :$LATEGAME_SHOWDOWN_PORT" >&2
tail -20 /tmp/lg_setup_showdown.log >&2
exit 1

#!/usr/bin/env bash
# One-time cluster setup: conda (bootstrapped if absent) + the lategame env + a built Showdown.
#
# MEASURED on UMIACS Nexus (not guessed): `module avail`, `which conda`, and every common install
# path (~/miniconda3, ~/anaconda3, /opt/conda, ...) all come up EMPTY. There is no conda anywhere
# on the login node. So this does not try to locate one -- it bootstraps a private Miniforge3
# under $REPO_DIR/.miniforge3 (no root needed, self-contained, gitignored). Everything else here
# (the port override, the server handshake, the job scripts) is tested locally.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
CONDA_ROOT="$REPO_DIR/.miniforge3"

echo "== node =="
# Showdown needs node >=16 (third_party/pokemon-showdown/package.json). Prefer a cluster module --
# on Nexus `module load nodejs` alone already satisfies this (measured: v16.20.2). If the conda env
# below pulls a newer nodejs (environment.yml wants >=18), `conda activate` puts it first on PATH
# and it takes over harmlessly; nothing here depends on which one wins.
if command -v module >/dev/null 2>&1; then
  module load nodejs 2>/dev/null || module load node 2>/dev/null || \
    echo "  no nodejs module found -- will rely on conda's"
fi
command -v node >/dev/null 2>&1 && echo "  $(node --version) (pre-conda)"

echo "== conda =="
if command -v conda >/dev/null 2>&1; then
  echo "  found on PATH: $(command -v conda)"
  CONDA_BASE="$(conda info --base)"
elif [ -x "$CONDA_ROOT/bin/conda" ]; then
  echo "  found a previous bootstrap at $CONDA_ROOT"
  CONDA_BASE="$CONDA_ROOT"
else
  echo "  none found -- bootstrapping Miniforge3 into $CONDA_ROOT"
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64)         INSTALLER=Miniforge3-Linux-x86_64.sh ;;
    aarch64|arm64)  INSTALLER=Miniforge3-Linux-aarch64.sh ;;
    *) echo "  unsupported architecture '$ARCH' -- install conda manually" >&2; exit 1 ;;
  esac
  FETCH="curl -fsSL"
  command -v curl >/dev/null 2>&1 || FETCH="wget -qO-"
  TMP_INSTALLER="$(mktemp)"
  $FETCH "https://github.com/conda-forge/miniforge/releases/latest/download/$INSTALLER" > "$TMP_INSTALLER"
  bash "$TMP_INSTALLER" -b -p "$CONDA_ROOT"
  rm -f "$TMP_INSTALLER"
  CONDA_BASE="$CONDA_ROOT"
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
echo "  $(conda --version)"

echo "== conda env =="
if conda env list | grep -q '^lategame '; then
  echo "  'lategame' exists; updating from environment.yml"
  conda env update -n lategame -f environment.yml --prune
else
  conda env create -f environment.yml
fi
conda activate lategame
echo "  active env node: $(node --version)"

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

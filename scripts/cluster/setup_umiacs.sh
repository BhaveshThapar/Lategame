#!/usr/bin/env bash
# One-time cluster setup: conda (bootstrapped if absent) + the rotomai env + a built Showdown.
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
  # A NAMED .sh file in its own directory, NOT a bare extensionless `mktemp` path: the
  # constructor-based Miniforge/Miniconda installer self-checks whether it is being sourced by
  # comparing $0 to BASH_SOURCE, and that check is known to misfire on an extensionless temp
  # file -- especially when /tmp is itself a symlinked or bind-mounted path, which some HPC
  # systems use. A clean, named, non-symlinked path avoids it.
  INSTALL_TMP_DIR="$(mktemp -d)"
  INSTALLER_PATH="$INSTALL_TMP_DIR/$INSTALLER"
  $FETCH "https://github.com/conda-forge/miniforge/releases/latest/download/$INSTALLER" > "$INSTALLER_PATH"
  chmod +x "$INSTALLER_PATH"
  if ! bash "$INSTALLER_PATH" -b -p "$CONDA_ROOT"; then
    echo >&2
    echo "  FATAL: the Miniforge installer refused to run." >&2
    echo "  If it printed \"Please run using bash/dash/sh/zsh, but not '.' or 'source'\", that" >&2
    echo "  is a known installer quirk (its sourced-vs-executed self-check misfiring), not this" >&2
    echo "  script actually sourcing anything. Try running it by hand once:" >&2
    echo "    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/$INSTALLER -o \$HOME/mf.sh" >&2
    echo "    chmod +x \$HOME/mf.sh && bash \$HOME/mf.sh -b -p \"$CONDA_ROOT\"" >&2
    exit 1
  fi
  rm -rf "$INSTALL_TMP_DIR"
  CONDA_BASE="$CONDA_ROOT"
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
echo "  $(conda --version)"

echo "== conda env =="
if conda env list | grep -q '^rotomai '; then
  echo "  'rotomai' exists; updating from environment.yml"
  conda env update -n rotomai -f environment.yml --prune
else
  conda env create -f environment.yml
fi
conda activate rotomai
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
export ROTOMAI_SHOWDOWN_PORT=8199
bash scripts/run_server.sh "$ROTOMAI_SHOWDOWN_PORT" > /tmp/ra_setup_showdown.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  if python -c "
import socket,sys
s=socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(('127.0.0.1', $ROTOMAI_SHOWDOWN_PORT)) == 0 else 1)" 2>/dev/null; then
    echo "  OK -- Showdown answered on :$ROTOMAI_SHOWDOWN_PORT"
    python -c "
from rotomai.config import LOCAL_SERVER
assert ':8199/' in LOCAL_SERVER.websocket_url, LOCAL_SERVER
print('  OK -- rotomai.config points at', LOCAL_SERVER.websocket_url)"
    echo
    echo "setup complete. next:  sbatch scripts/cluster/stage_a.slurm"
    exit 0
  fi
  sleep 2
done
echo "FAILED: Showdown never opened :$ROTOMAI_SHOWDOWN_PORT" >&2
tail -20 /tmp/ra_setup_showdown.log >&2
exit 1

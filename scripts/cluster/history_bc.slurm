#!/usr/bin/env bash
# Stage 1+2 of the history gate: BC then AWR, one array task per (arm, seed).
#
# Pre-registered in results/history_bc_prereg.json. Read it before submitting -- in particular the
# BLOCKING PRECONDITION: scripts/bc_shard_fidelity_gate.py must return PASS before any of this runs.
# The shard is a RECONSTRUCTION (the original is off disk and cannot be re-fetched without
# confounding the comparison), and training on an unverified reconstruction is the one failure the
# whole no-re-fetch argument exists to avoid.
#
#   python scripts/bc_shard_fidelity_gate.py            # must print PASS
#   python scripts/rebuild_bc_shard.py --out data/gen9ou_v7_bc_rebuilt.npz
#   BUILD=v30h sbatch -p tron --qos=medium --array=0-5 scripts/cluster/history_bc.slurm
#
# SIX TASKS, TWO ARMS. Array index 0-2 is the CONTROL (history 0) at seeds 0/1/2; 3-5 is the ARM
# (history 4) at the same seeds. Both arms in one array on purpose: they then share a submission,
# a node pool and a queue position, so an arm cannot be advantaged by running on a quieter day.
# This is the same reason strength_gate.slurm scores both arms in one invocation.
#
# NO SHOWDOWN SERVER IS STARTED. BC and AWR are offline -- they read a shard and write a
# checkpoint. That also means no port to allocate, so this wrapper is the one thing in
# scripts/cluster/ that cannot collide with a concurrent job over ws://localhost.
#
# Only 4 tasks run concurrently (tron caps a user at cpu=32,mem=256G and each task asks 8/64G), so
# budget in TASK-HOURS / 4: 6 tasks x ~2-4 h = ~12-24 task-hours, i.e. ~4-7 h wall-clock. The
# history arm is slower per step -- 5 frames means 5x the tokens through the encoder -- and that
# cost is REPORTED, not adjudicated (see the pre-registration).
#
#SBATCH --job-name=ra-histbc
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/slurm-%x-%A_%a.out
# NOTE: no --gres=gpu. The net is 4.56M parameters and RAM is the binding constraint.

set -euo pipefail
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$COMMON_DIR/_job_common.sh" ]] || COMMON_DIR="${SLURM_SUBMIT_DIR:-$COMMON_DIR}/scripts/cluster"
source "$COMMON_DIR/_job_common.sh"

BUILD="${BUILD:?set BUILD, e.g. BUILD=v30h sbatch --array=0-5 ...}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
SEEDS_PER_ARM="${SEEDS_PER_ARM:-3}"

# Arm from the array index, so ONE submission covers both and neither can be re-run alone by
# accident -- a re-run arm measured against a control from another day is the cross-run confound
# this project has already measured at 0.027.
if (( TASK < SEEDS_PER_ARM )); then
  ARM="control"; HISTORY=0; SEED="$TASK"
else
  ARM="hist"; HISTORY="${HISTORY_K:-4}"; SEED="$(( TASK - SEEDS_PER_ARM ))"
fi

BC_SHARD="${BC_SHARD:-data/gen9ou_v7_bc_rebuilt.npz}"
RL_SHARD="${RL_SHARD:-data/gen9ou_v7_rl.npz}"
BC_EPOCHS="${BC_EPOCHS:-20}"
RL_EPOCHS="${RL_EPOCHS:-30}"

BC_OUT="checkpoints/bc_${BUILD}_${ARM}_s${SEED}.pt"
RL_OUT="checkpoints/offrl_${BUILD}_${ARM}_s${SEED}.pt"
RUN_LOG="logs/history/${BUILD}/${ARM}_s${SEED}.log"
mkdir -p "$(dirname "$RUN_LOG")"

# Preflight: name the missing file NOW, not forty minutes in. /data/ and /checkpoints/ are
# gitignored, so a fresh node has neither.
[[ -f "$BC_SHARD" ]] || {
  echo "[job] FATAL: BC shard missing: $BC_SHARD"
  echo "[job]   python scripts/rebuild_bc_shard.py --out $BC_SHARD"; exit 2; }
[[ -f "$RL_SHARD" ]] || { echo "[job] FATAL: RL shard missing: $RL_SHARD"; exit 2; }

# The reconstruction must have been verified, and the verification must be on disk. A PASS that
# nobody can point at afterwards is not a pre-registration, it is a memory.
FIDELITY="results/bc_shard_fidelity_gate.json"
[[ -f "$FIDELITY" ]] || {
  echo "[job] FATAL: $FIDELITY missing -- the shard reconstruction was never verified."
  echo "[job]   python scripts/bc_shard_fidelity_gate.py"; exit 2; }
python - "$FIDELITY" <<'PY'
import json, sys
verdict = json.load(open(sys.argv[1]))["verdict"]
if verdict != "PASS":
    sys.exit(f"[job] FATAL: BC shard fidelity gate says {verdict}; this arm must not run.")
print(f"[job] shard fidelity: {verdict}")
PY

echo "[job] BUILD=$BUILD arm=$ARM history=$HISTORY seed=$SEED"

{
  python -u -m rotomai.cli train \
    --data "$BC_SHARD" \
    --out  "$BC_OUT" \
    --model-type entity_transformer \
    --id-embed-init prior \
    --history "$HISTORY" \
    --epochs "$BC_EPOCHS" \
    --seed "$SEED" \
    --device cpu

  # --history MUST match the BC init's: the AWR warm-start does a STRICT state-dict load, so a
  # mismatch fails on `time_embed` rather than silently training a sequence critic on flat frames.
  python -u -m rotomai.cli train-rl \
    --data "$RL_SHARD" \
    --out  "$RL_OUT" \
    --model entity_transformer \
    --id-embed-init prior \
    --history "$HISTORY" \
    --bc-init "$BC_OUT" \
    --epochs "$RL_EPOCHS" \
    --seed "$SEED" \
    --device cpu
} 2>&1 | tee -a "$RUN_LOG"

echo "[job] $ARM seed $SEED done -> $RL_OUT"
echo "[job] stage 3 (PPO) reuses the existing wrapper, one array per arm, distinct port bases:"
echo "[job]   BUILD=${BUILD}c INIT=checkpoints/offrl_${BUILD}_control_s0.pt \\"
echo "[job]     ROTOMAI_SHOWDOWN_PORT_BASE=8600 sbatch --array=0-2 scripts/cluster/ppo_seed.slurm"
echo "[job]   BUILD=${BUILD}h INIT=checkpoints/offrl_${BUILD}_hist_s0.pt \\"
echo "[job]     ROTOMAI_SHOWDOWN_PORT_BASE=8700 sbatch --array=0-2 scripts/cluster/ppo_seed.slurm"

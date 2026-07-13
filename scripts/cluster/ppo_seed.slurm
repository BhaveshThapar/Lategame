#!/usr/bin/env bash
# One PPO seed per array task, each with its own Showdown server.
#
# On the laptop the three seeds ran SEQUENTIALLY (4.6 h) because they shared one server on the
# hardcoded port 8000. With LATEGAME_SHOWDOWN_PORT they are embarrassingly parallel: wall-clock
# collapses to a single seed's runtime.
#
#   sbatch --array=0-2 scripts/cluster/ppo_seed.slurm
#
# Each seed writes results/ppo_ou_gate_<BUILD>_s{N}.json. Pool them AFTERWARDS with
# scripts/merge_gate_seeds.py -- never hand a single-seed file to seed_strength_gate.py, which
# will happily pool 300 instead of 900 and print a verdict that has lost the power it was
# designed to buy.
#
#SBATCH --job-name=lg-ppo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.out
# NOTE: no --gres=gpu (4.56M params). NOTE: set --partition for UMIACS.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_job_common.sh"

SEED="${SLURM_ARRAY_TASK_ID:-0}"
BUILD="${BUILD:?set BUILD, e.g. BUILD=v22 sbatch --array=0-2 ...}"
INIT="${INIT:?set INIT, the warm-start checkpoint}"
PREFIX="${PREFIX:-ppo_${BUILD}}"

start_showdown

python -u scripts/ppo_continue_gate.py \
  --init          "$INIT" \
  --ckpt-prefix   "$PREFIX" \
  --out           "results/ppo_ou_gate_${BUILD}_s${SEED}.json" \
  --seeds         "$SEED" \
  --format        gen9ou \
  --team-pool     lategame/teambuilding/data/teams_gen9ou.packed \
  --loop-penalty  4.0 \
  --iters         50 \
  --games-per-opp 48 \
  --eval-n        100 \
  --ladder-n      20 \
  --ent-coef      0.01 --ent-coef-final 0.0 \
  --lr            2.5e-4 --lr-final 5e-5 \
  2>&1 | tee "ppo_${BUILD}_s${SEED}.log"

# The stdout log is gitignored (*.log) and is the ONLY source of the trust-region certificate --
# ppo_continue_gate's JSON stores just the win-rate curve. Persist it NOW, in-job: a NULL is only
# attributable to the lever if the trust region did not bind, and Build 20 nearly lost this.
python -u scripts/ppo_telemetry.py \
  --log "$SEED" "ppo_${BUILD}_s${SEED}.log" \
  --out "results/ppo_ou_telemetry_${BUILD}_s${SEED}.json"

echo "[job] seed ${SEED} done"

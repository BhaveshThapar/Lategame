#!/usr/bin/env bash
# One PPO seed per array task, each with its own Showdown server.
#
# On the laptop the three seeds ran SEQUENTIALLY (4.6 h) because they shared one server on the
# hardcoded port 8000. With LATEGAME_SHOWDOWN_PORT they are embarrassingly parallel: wall-clock
# collapses to a single seed's runtime.
#
#   sbatch --array=0-2 scripts/cluster/ppo_seed.slurm
#
# ITERS and TARGET_KL default to the v17-v22 values, so an unset submit reproduces those builds
# exactly. Build 23 raises both (ITERS=80 TARGET_KL=0.06) and runs TWO arms concurrently -- the
# second needs LATEGAME_SHOWDOWN_PORT_BASE=8200 or the arms share Showdown servers (_job_common.sh).
#
# Build 24 adds ANNEAL_ITERS, because ITERS silently doubles as the lr/ent anneal horizon: at
# iteration 40, v22 (ITERS=50) sat at lr 9.08e-05 / ent 0.0020 while v23b (ITERS=80) sat at
# 1.51e-04 / 0.0051, so raising ITERS is THREE changes, not one. ANNEAL_ITERS=50 with ITERS=80
# holds both schedules at their finals past iteration 50, isolating update count. Its 2x2 submits
# FOUR arrays, so give each its own base: 8200 / 8300 / 8400 / 8500.
#
# THEY DO NOT ALL RUN AT ONCE, and the estimate must not assume they do. The tron QoS caps a user
# at cpu=32,mem=256G and each task here asks 8 CPU / 64 GB, so EXACTLY 4 TASKS run concurrently --
# measured on Build 23's own sacct, where 7181042_1 started 00:51:10 against 7181042_0's 00:51:09
# end, and 7181042_2 at 03:00:38 against 7181039_1's 03:00:38 end. Slurm backfills the instant a
# slot frees, so submit all the arrays at once and do NOT stage waves by hand -- but budget in
# TASK-HOURS / 4. Build 24: 12 tasks, ~60 task-hours, ~15-18 h wall-clock.
#
# Each seed writes results/ppo_ou_gate_<BUILD>_s{N}.json. Pool them AFTERWARDS with
# scripts/merge_gate_seeds.py -- never hand a single-seed file to seed_strength_gate.py, which
# will happily pool 300 instead of 900 and print a verdict that has lost the power it was
# designed to buy.
#
#SBATCH --job-name=lg-ppo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=slurm-%x-%A_%a.out
# NOTE: no --gres=gpu (4.56M params). NOTE: set --partition for UMIACS.
# 20h, not 12: v22 took 4:50-5:24 for 50 iters, and Build 23 runs 80 at a looser KL bar -- fewer
# early stops means more full 4-epoch updates, so ~9-11h. medium QoS allows 2 days; headroom is free.

set -euo pipefail
# sbatch runs a spool COPY of this script, so ${BASH_SOURCE[0]}'s sibling _job_common.sh is NOT
# beside it at run time. Try the sibling first (direct `bash` runs), else fall back to the submit
# dir -- the repo root, per scripts/cluster/README.md's documented `sbatch` cwd.
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$COMMON_DIR/_job_common.sh" ]] || COMMON_DIR="${SLURM_SUBMIT_DIR:-$COMMON_DIR}/scripts/cluster"
source "$COMMON_DIR/_job_common.sh"

SEED="${SLURM_ARRAY_TASK_ID:-0}"
BUILD="${BUILD:?set BUILD, e.g. BUILD=v22 sbatch --array=0-2 ...}"
INIT="${INIT:?set INIT, the warm-start checkpoint}"
PREFIX="${PREFIX:-ppo_${BUILD}}"
ITERS="${ITERS:-50}"
TARGET_KL="${TARGET_KL:-0.03}"
# Build 24: the lr/ent anneal horizon. Unset == ITERS, which reproduces v17-v23. Set it to hold
# both schedules at their finals past that iteration, which is what separates "more updates" from
# "a slower anneal" -- raising ITERS alone moves both at once.
ANNEAL_ITERS="${ANNEAL_ITERS:-}"
# The bar the KL early-stop actually enforces (ppo.py:212). Derived here, ONCE, so the run and its
# certificate cannot disagree about what the trust region was.
KL_BAR="$(python -c "print(1.5 * ${TARGET_KL})")"

# Built as an array, not interpolated: an unset ANNEAL_ITERS must pass NO flag at all, so the
# default path is byte-identical to v17-v23. `if`, not `&&`, because a false `&&` as a bare
# statement trips `set -e`.
ANNEAL_ARGS=()
if [[ -n "$ANNEAL_ITERS" ]]; then
  ANNEAL_ARGS=(--anneal-iters "$ANNEAL_ITERS")
fi

start_showdown

python -u scripts/ppo_continue_gate.py \
  --init          "$INIT" \
  --ckpt-prefix   "$PREFIX" \
  --out           "results/ppo_ou_gate_${BUILD}_s${SEED}.json" \
  --seeds         "$SEED" \
  --format        gen9ou \
  --team-pool     lategame/teambuilding/data/teams_gen9ou.packed \
  --loop-penalty  4.0 \
  --iters         "$ITERS" \
  --games-per-opp 48 \
  --eval-n        100 \
  --ladder-n      20 \
  --ent-coef      0.01 --ent-coef-final 0.0 \
  --lr            2.5e-4 --lr-final 5e-5 \
  --target-kl     "$TARGET_KL" \
  ${ANNEAL_ARGS[@]+"${ANNEAL_ARGS[@]}"} \
  2>&1 | tee "ppo_${BUILD}_s${SEED}.log"

# The stdout log is gitignored (*.log) and is the ONLY source of the trust-region certificate --
# ppo_continue_gate's JSON stores just the win-rate curve. Persist it NOW, in-job: a NULL is only
# attributable to the lever if the trust region did not bind, and Build 20 nearly lost this.
# --kl-bar tracks TARGET_KL: a certificate that named 0.045 for a run at 0.09 would claim a bar the
# optimizer never enforced, in the one artifact a NULL's attribution rests on.
python -u scripts/ppo_telemetry.py \
  --log "$SEED" "ppo_${BUILD}_s${SEED}.log" \
  --kl-bar "$KL_BAR" \
  --out "results/ppo_ou_telemetry_${BUILD}_s${SEED}.json"

echo "[job] seed ${SEED} done"

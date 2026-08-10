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
# 64G is a HARD CEILING, not a choice: `sacctmgr show qos medium` sets MaxTRES=cpu=8,mem=64G PER
# JOB, so `--mem` above 64G is rejected at submit time ("Memory required by task is not available")
# rather than queued. high allows 128G but caps walltime at 24h; default allows only 4 CPU / 32G.
#
# THIS BOUNDS HOW LONG AN ARM CAN BE. RSS scales ~linearly with iters -- sacct MaxRSS is 40.7 GiB
# at 160 (Build 25, 7188055) and 59.3 GiB at 240 (v26a, 7215446), i.e. ~0.23 GiB/iter, projecting
# ~78 GiB at 320. So 240 iters fits with ~5 GiB to spare and 320 does NOT: v26b's first attempt was
# OUT_OF_MEMORY at 304 iters, having reached exactly this cap. Anything past ~260 iters needs the
# resume path (a fresh process resets RSS), not a bigger --mem.
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/slurm/slurm-%x-%A_%a.out
# NOTE: no --gres=gpu (4.56M params). NOTE: set --partition for UMIACS.
#
# 36h, raised from 20h for Build 26. Build 25's sacct (7188055) is the measurement to budget off:
# v25b's 160 iters took 11:03:35 / 11:23:20 / 11:30:31, i.e. ~4.15-4.3 min/iter -- FASTER than the
# 4.8-5.3 plan.md assumed. That projects 240 iters -> ~17h and 320 iters -> ~23h, so the 320-iter
# arm blew the old 20h limit outright and 240 was close enough that a slow seed would too.
#
# Note that Build 25 only survived because it was submitted with `--time=24:00:00` on the command
# line: sacct reports Timelimit=1-00:00:00 against this file's then-20h, i.e. the script and the
# real submission had already drifted apart. Raising it here is what stops that recurring.
#
# The headroom matters more than usual because ppo_continue_gate.py has NO RESUME: it runs from
# --init for `iters` iterations with no restart path, so a walltime kill loses the whole arm --
# ~23h of it for a 320-iter seed. medium QoS allows 2 days (sacctmgr: medium MaxWall=2-00:00:00),
# so 36h is free and leaves ~13h of margin on the longest arm planned.

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

# The run log, named ONCE. It is the only source of the trust-region certificate (see below), and
# the `tee` that writes it and the `ppo_telemetry.py` that reads it used to repeat the literal
# path -- a drift between the two would have silently produced an empty certificate.
RUN_LOG="logs/ppo/${BUILD}/ppo_${BUILD}_s${SEED}.log"
mkdir -p "$(dirname "$RUN_LOG")"

# Built as an array, not interpolated: an unset ANNEAL_ITERS must pass NO flag at all, so the
# default path is byte-identical to v17-v23. `if`, not `&&`, because a false `&&` as a bare
# statement trips `set -e`.
ANNEAL_ARGS=()
if [[ -n "$ANNEAL_ITERS" ]]; then
  ANNEAL_ARGS=(--anneal-iters "$ANNEAL_ITERS")
fi

# Build 26: RESUME=1 continues from checkpoints/<PREFIX>_s<SEED>/'s last completed iteration.
# Needed because RSS grows ~0.23 GiB/iter against medium's hard 64 GB per-job cap, so an arm past
# ~260 iters cannot run in ONE process however much walltime it gets -- a fresh process is what
# resets the memory. Unset == start fresh, byte-identical to Build 25.
RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARGS=(--resume)
fi

# The run log is append-on-resume: truncating it would throw away the first chunk's trust-region
# certificate, which ppo_telemetry.py reads and without which a NULL is unattributable.
TEE_ARGS=(-a)

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
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  2>&1 | tee "${TEE_ARGS[@]}" "$RUN_LOG"

# The stdout log is gitignored (*.log) and is the ONLY source of the trust-region certificate --
# ppo_continue_gate's JSON stores just the win-rate curve. Persist it NOW, in-job: a NULL is only
# attributable to the lever if the trust region did not bind, and Build 20 nearly lost this.
# --kl-bar tracks TARGET_KL: a certificate that named 0.045 for a run at 0.09 would claim a bar the
# optimizer never enforced, in the one artifact a NULL's attribution rests on.
python -u scripts/ppo_telemetry.py \
  --log "$SEED" "$RUN_LOG" \
  --kl-bar "$KL_BAR" \
  --out "results/ppo_ou_telemetry_${BUILD}_s${SEED}.json"

echo "[job] seed ${SEED} done"

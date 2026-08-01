#!/usr/bin/env bash
# Re-probe an EXISTING checkpoint's |G|^2(0) at a different probe seed.
#
# Why this exists: plan.md 13.1 records that cos read 0.182 on one probe and 0.429 on a RERUN of
# the same checkpoint -- the cosine is noisy ACROSS PROBE RUNS, not just across checkpoints. Any
# gate decided by cos clearing the 0.3 floor by a small margin therefore needs a second seed
# before it is spent on an expensive build.
#
# Everything except --seed matches the Stage A / B-0 protocol exactly, so the reruns are
# comparable: --games-per-opp 48, --rollouts 6, --splits 5 (--splits DEFAULTS TO 20; passing 5
# is mandatory), --policy == --init to probe the warm start itself.
#
#   CKPT=checkpoints/offrl_gen9ou_wide_e10_s0.pt TAG=b22_stageB0_e10 PROBE_SEED=1 \
#     LATEGAME_SHOWDOWN_PORT=8200 sbatch -p tron --qos=medium scripts/cluster/probe_replicate.slurm
#
# PORT: this job has no array index, so _job_common.sh would default it to 8100 -- which COLLIDES
# with a concurrent `--array=0-2` job's task 0. Two jobs sharing one Showdown silently battle into
# each other's games. Always pin LATEGAME_SHOWDOWN_PORT clear of any running array.
#
#SBATCH --job-name=lg-probe-rep
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
# NOTE: no --gres=gpu. NOTE: needs -p tron --qos=medium (default QoS caps CPUs at 4).

set -euo pipefail
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$COMMON_DIR/_job_common.sh" ]] || COMMON_DIR="${SLURM_SUBMIT_DIR:-$COMMON_DIR}/scripts/cluster"
source "$COMMON_DIR/_job_common.sh"

CKPT="${CKPT:?set CKPT, the checkpoint to re-probe}"
TAG="${TAG:?set TAG, e.g. b22_stageB0_e10 -- names the output file}"
PROBE_SEED="${PROBE_SEED:-1}"
OUT="results/grad_noise_diag_${TAG}_seed${PROBE_SEED}.json"

[[ -f "$CKPT" ]] || { echo "[job] FATAL: checkpoint not found at $CKPT" >&2; exit 1; }
[[ -e "$OUT" ]] && { echo "[job] FATAL: $OUT exists; refusing to overwrite a probe result" >&2; exit 1; }

echo "[job] re-probing $CKPT at --seed $PROBE_SEED on port $LATEGAME_SHOWDOWN_PORT -> $OUT"
start_showdown

python -u scripts/grad_noise_diag.py \
  --policy        "$CKPT" \
  --init          "$CKPT" \
  --games-per-opp 48 \
  --rollouts      6 \
  --splits        5 \
  --seed          "$PROBE_SEED" \
  --out           "$OUT"

echo "[job] replication done -> $OUT"
echo "[job] compare cos_at_budget and noise_scale.g_norm_sq against the seed-0 probe."

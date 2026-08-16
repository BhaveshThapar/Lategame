#!/usr/bin/env bash
# Build 22 STAGE A -- the pre-registered discriminator (docs/RESULTS.md).
#
# Probe |G|^2 at iter_0 (the WARM START, before any PPO) for both nets. H22 says the wide net is
# BORN flat -- that the offline objective, not PPO, manufactures the stationary point.
#
#   wide  |G|^2(0) ~ 0.057 (its own iter_10 value)  => BORN FLAT, H22 confirmed
#   wide  |G|^2(0) >> 0.057                         => collapses DURING PPO, H22 refuted
#   narrow|G|^2(0) not ~ 4.540 (its iter_10 value)  => the instrument is wrong; conclude nothing
#
# The narrow arm is the CONTROL. Read the WITHIN-architecture trend: absolute |G|^2 is not
# comparable across widths (gradient norms scale with parameterization); cos and B_simple are.
#
#SBATCH --job-name=lg-b22-stageA
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/slurm-%x-%A_%a.out
# NOTE: no --gres=gpu. The net is 4.56M params; a GPU buys nothing and lengthens the queue.
# NOTE: --partition is intentionally unset -- fill in whatever UMIACS gives you.

set -euo pipefail
# sbatch runs a spool COPY of this script, so ${BASH_SOURCE[0]}'s sibling _job_common.sh is NOT
# beside it at run time. Try the sibling first (direct `bash` runs), else fall back to the submit
# dir -- the repo root, per scripts/cluster/README.md's documented `sbatch` cwd.
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$COMMON_DIR/_job_common.sh" ]] || COMMON_DIR="${SLURM_SUBMIT_DIR:-$COMMON_DIR}/scripts/cluster"
source "$COMMON_DIR/_job_common.sh"

# Array task 0 = wide (the net under test), 1 = narrow (the control).
case "$SLURM_ARRAY_TASK_ID" in
  0) CKPT=checkpoints/offrl_gen9ou_wide_s0.pt ; TAG=wide   ;;
  1) CKPT=checkpoints/offrl_gen9ou_v7_s0.pt   ; TAG=narrow ;;
  *) echo "unexpected array index" >&2; exit 1 ;;
esac

start_showdown

# --policy == --init: probe the warm start itself. _league_for handles k=0 -- the league is
# [init] + anchors, exactly the mix iteration 0 faced.
# gpo 48 = the budget BOTH v20 and v21 actually trained at. rollouts 6 / splits 5 match v20's
# probe (--splits DEFAULTS TO 20; passing 5 is mandatory for comparability).
python -u scripts/grad_noise_diag.py \
  --policy        "$CKPT" \
  --init          "$CKPT" \
  --games-per-opp 48 \
  --rollouts      6 \
  --splits        5 \
  --out           "results/grad_noise_diag_b22_stageA_${TAG}.json"

echo "[job] Stage A ($TAG) done -> results/grad_noise_diag_b22_stageA_${TAG}.json"

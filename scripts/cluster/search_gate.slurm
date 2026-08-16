#!/usr/bin/env bash
# Build 27 Gate B: does depth-2 search compound on a format that HAS headroom?
#
# L11-L14 retired test-time search at parity -- all of it measured on gen9-RB, which Lever 15
# then proved FORMAT_BOUND (near-optimal search there reaches 0.500 because NOTHING beats the
# heuristic). This re-runs the strongest arm (depth-2 expectimax, white-box opponent model) on
# gen9ou, where headroom is proven and the base is 0.74.
#
# SHARDED, BECAUSE SEARCH IS SLOW AND ITS DRIVER IS SERIAL. `SearchAgent` holds ONE
# `ForwardModel` node subprocess and every node RPC queues on that single pipe, so raising
# --concurrency buys nothing. Measured: ~29 s per search battle. The pre-registered n=2500/arm
# needs ~5,000 search battles => ~40 h serial, so it is split across an array and pooled by
# scripts/merge_search_shards.py.
#
#   N_PER_SHARD=250 sbatch -p tron --qos=medium --array=0-9 scripts/cluster/search_gate.slurm
#
# PORT BASE: 8600, clear of PPO (8100-8202), eval_ladder (8300), build26_analysis (8400) and the
# interactive 8500. Two concurrent arrays MUST use different bases -- see _job_common.sh.
#
#SBATCH --job-name=lg-search
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/slurm-%x-%A_%a.out
# NOTE: no --gres=gpu -- inference is CPU-cheap; the node forward model is the bottleneck.

set -euo pipefail
export LATEGAME_SHOWDOWN_PORT_BASE="${LATEGAME_SHOWDOWN_PORT_BASE:-8600}"

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$COMMON_DIR/_job_common.sh" ]] || COMMON_DIR="${SLURM_SUBMIT_DIR:-$COMMON_DIR}/scripts/cluster"
source "$COMMON_DIR/_job_common.sh"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
N_PER_SHARD="${N_PER_SHARD:-250}"
SANITY_N="${SANITY_N:-40}"
FMT="${FMT:-gen9ou}"
INIT="${INIT:-checkpoints/ppo_v26b_s0/iter_300.pt}"
TEAM_POOL="${TEAM_POOL:-lategame/teambuilding/data/teams_gen9ou.packed}"

# DOUBLES: this same job is the M2 leg of the VGC ceiling probe, and two settings are not optional.
#
#   FMT=gen9vgc2025regi INIT=checkpoints/doubles_offrl_vgc_v2.pt \
#   TEAM_POOL=lategame/teambuilding/data/teams_gen9vgc.packed \
#   N_PER_SHARD=30 SANITY_N=10 OUT_PREFIX=results/rpredict_search_vgc \
#   LATEGAME_SHOWDOWN_PORT_BASE=8700 sbatch -p tron --qos=medium --array=0-9 \
#     scripts/cluster/search_gate.slurm
#
# 1. `search` is in `_SINGLES_ONLY_AGENTS` and `build_player` refuses it on a doubles format --
#    UNLESS LATEGAME_SEARCH_SHAPED_ONLY=1, where the leaf is `data.reward.state_value` (reads the
#    battle object, no singles encoder), the driver enumerates JOINT orders, and the fallback is
#    the doubles-capable heuristic rule. Set here rather than left to the caller's environment,
#    because forgetting it is a hard refusal at build time and an hour of queue wasted.
# 2. LOOP_PENALTY must be 0. The singles default of 4 builds a 26-wide `LoopGuard` penalizing
#    indices 0-5; on the doubles layout index 0 is PASS, so it would penalize passing and five of
#    six switches -- not a weaker guard, a differently-wrong one.
case "$FMT" in
  *vgc*|*doubles*)
    export LATEGAME_SEARCH_SHAPED_ONLY=1
    LOOP_PENALTY="${LOOP_PENALTY:-0}"
    echo "[job] doubles format $FMT: shaped-only search leaf, loop_penalty $LOOP_PENALTY"
    ;;
esac
LOOP_PENALTY="${LOOP_PENALTY:-4}"
DEPTH="${DEPTH:-2}"
ARMS="${ARMS:-whitebox}"
OUT_PREFIX="${OUT_PREFIX:-results/rpredict_search_ou}"
OUT="${OUT:-${OUT_PREFIX}_shard${SHARD}.json}"

# Every shard must exist before anything is pooled, so fail loudly on a missing checkpoint here
# rather than after hours of battles.
[[ -f "$INIT" ]] || { echo "[job] FATAL: checkpoint $INIT not found"; exit 2; }
[[ "$FMT" == *random* ]] || [[ -f "$TEAM_POOL" ]] || {
  echo "[job] FATAL: teambuilt format $FMT needs TEAM_POOL, missing: $TEAM_POOL"; exit 2; }

ARGS=(--gate b --arms "$ARMS" --format "$FMT" --init "$INIT" --depth "$DEPTH"
      --n "$N_PER_SHARD" --sanity-n "$SANITY_N" --concurrency 1
      --seed "$SHARD" --loop-penalty "$LOOP_PENALTY" --out "$OUT")
[[ "$FMT" == *random* ]] || ARGS+=(--team-pool "$TEAM_POOL")

start_showdown

echo "[job] search gate shard $SHARD: ${ARGS[*]}"
python -u scripts/rpredict_oppmodel_gate.py "${ARGS[@]}"

echo "[job] shard $SHARD done -> $OUT"

#!/usr/bin/env bash
# R-EVAL: the agent-only eval ladder (lategame eval-ladder).
#
# Rates a FIELD of agents against each other rather than one agent against a fixed baseline. That
# is the only condition under which GXE/Glicko-1 carry information a win rate did not -- against a
# single pinned opponent they are a reparameterisation of it (lategame/eval/rating.py says so at
# length). This is what plan.md 12's "private/agent-only server or eval ladder" asks for.
#
# NOT a replacement for scripts/cluster/strength_gate.slurm. That remains the authoritative
# build-vs-build statistic; this answers a different question (where does a build sit in a field).
#
#   OUT=results/eval_ladder_gen9ou.json sbatch -p tron --qos=medium scripts/cluster/eval_ladder.slurm
#
#   FIELD=random,heuristic,offrl@checkpoints/ppo_v25b_s0/iter_160.pt \
#   N=300 OUT=results/eval_ladder_custom.json \
#     sbatch -p tron --qos=medium scripts/cluster/eval_ladder.slurm
#
# THE PORT BASE IS NOT OPTIONAL WHEN ANOTHER BUILD IS IN FLIGHT. _job_common.sh computes
# 8100 + TASK_ID, and a non-array job takes TASK_ID=0 -> port 8100, which is exactly the port
# `ppo_seed.slurm`'s array task 0 claims. Colliding jobs on one node SILENTLY SHARE a Showdown
# server rather than failing, so they see each other's battles. The default below moves this job
# clear of both bases a PPO build uses (8100-8102 and the 8200-8202 second array).
#
#SBATCH --job-name=lg-ladder
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
# NOTE: no --gres=gpu -- inference is CPU-cheap and the local server is the bottleneck.
# NOTE: needs -p tron --qos=medium (the default QoS caps CPUs at 4).

set -euo pipefail
export LATEGAME_SHOWDOWN_PORT_BASE="${LATEGAME_SHOWDOWN_PORT_BASE:-8300}"

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$COMMON_DIR/_job_common.sh" ]] || COMMON_DIR="${SLURM_SUBMIT_DIR:-$COMMON_DIR}/scripts/cluster"
source "$COMMON_DIR/_job_common.sh"

OUT="${OUT:?set OUT, the result json path}"
N="${N:-150}"
FMT="${FMT:-gen9ou}"
ANCHOR="${ANCHOR:-heuristic}"
CONCURRENCY="${CONCURRENCY:-20}"
TEAM_POOL="${TEAM_POOL:-lategame/teambuilding/data/teams_gen9ou.packed}"

ARGS=(--format "$FMT" --n "$N" --anchor "$ANCHOR" --concurrency "$CONCURRENCY" --out "$OUT")
[[ -n "${FIELD:-}" ]] && ARGS+=(--field "$FIELD")
# A teambuilt format needs a pool; Random Battles must NOT get one (the server supplies teams).
[[ "$FMT" == *randombattle* ]] || ARGS+=(--team-pool "$TEAM_POOL")
[[ -n "${LOOP_PENALTY:-}" ]] && ARGS+=(--loop-penalty "$LOOP_PENALTY")
[[ -n "${SAMPLE:-}" ]] && ARGS+=(--sample)

start_showdown

# The CLI preflights the anchor and every checkpoint BEFORE playing a battle, so a bad field fails
# in seconds rather than after an hour of games.
echo "[job] eval ladder: ${ARGS[*]}"
python -u -m lategame.cli eval-ladder "${ARGS[@]}"

echo "[job] eval ladder done -> $OUT"

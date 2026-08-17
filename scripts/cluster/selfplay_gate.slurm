#!/usr/bin/env bash
# M4 self-play as a BUILD: run the loop and leave a citable record behind.
#
# There was no cluster job for self-play at all. `rotomai selfplay` writes only
# <out_dir>/curve.json -- nothing under results/ -- so every M4 run before this one left an
# artifact the artifact checker cannot see and a later build cannot cite. The one bridge from the
# loop to results/ is scripts/curriculum_gate.py, which is what this job runs.
#
# ONE SEED PER ARRAY TASK, same contract as ppo_seed.slurm. The gate itself loops over --seeds,
# so a single task with --seeds 0,1,2 is also valid and is the right shape for a short capability
# record; the array is for the three-seed strength form.
#
#   # capability record (one seed, short) -- the FORMAT_BOUND branch
#   ARM=m4_vgc_cap FMT=gen9vgc2025regi INIT=checkpoints/doubles_offrl_vgc_v2.pt \
#   TEAM_POOL=rotomai/teambuilding/data/teams_gen9vgc.packed \
#   ITERS=2 GAMES_PER_OPP=24 EVAL_N=60 SEEDS=0 \
#   sbatch -p tron --qos=medium scripts/cluster/selfplay_gate.slurm
#
#   # strength arm (three seeds) -- the AMBIGUOUS / MODEL_BOUND branch
#   ARM=m4_vgc FMT=gen9vgc2025regi INIT=checkpoints/doubles_offrl_vgc_v2.pt \
#   TEAM_POOL=rotomai/teambuilding/data/teams_gen9vgc.packed \
#   SEEDS=0,1,2 sbatch -p tron --qos=medium scripts/cluster/selfplay_gate.slurm
#
# PORT BASE: 8800, clear of PPO (8100-8202), eval_ladder (8300), build26_analysis (8400), the
# interactive 8500, search (8600) and the VGC search array (8700). Two concurrent arrays MUST use
# different bases -- see _job_common.sh.
#
#SBATCH --job-name=ra-selfplay
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/slurm-%x-%A_%a.out
# NOTE: no --gres=gpu -- the fine-tune is small and the server is the bottleneck, as for ppo_seed.

set -euo pipefail
export ROTOMAI_SHOWDOWN_PORT_BASE="${ROTOMAI_SHOWDOWN_PORT_BASE:-8800}"

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$COMMON_DIR/_job_common.sh" ]] || COMMON_DIR="${SLURM_SUBMIT_DIR:-$COMMON_DIR}/scripts/cluster"
source "$COMMON_DIR/_job_common.sh"

ARM="${ARM:?set ARM, e.g. ARM=m4_vgc -- it names the record, checkpoints/<ARM>_s<seed>/ and data/}"
FMT="${FMT:-gen9randombattle}"
INIT="${INIT:?set INIT to the warm-start checkpoint}"
TEAM_POOL="${TEAM_POOL:-}"
SEEDS="${SEEDS:-0,1,2}"
ITERS="${ITERS:-12}"
GAMES_PER_OPP="${GAMES_PER_OPP:-48}"
EVAL_N="${EVAL_N:-200}"
LADDER_N="${LADDER_N:-300}"
MAX_BATTLE_TURNS="${MAX_BATTLE_TURNS:-}"
OUT="${OUT:-results/curriculum_gate_${ARM}.json}"

# Fail before the queue is spent, not after a full iteration of collection. Both of these are
# mistakes that otherwise surface hours in: a missing warm start dies at load, and a teambuilt
# format with no pool starts no battles while looking like it is running.
[[ -f "$INIT" ]] || { echo "[job] FATAL: warm-start checkpoint $INIT not found"; exit 2; }
[[ "$FMT" == *random* ]] || [[ -f "$TEAM_POOL" ]] || {
  echo "[job] FATAL: teambuilt format $FMT needs TEAM_POOL, missing: '$TEAM_POOL'"; exit 2; }

# A BC checkpoint clears the format and encoder checks and then dies on a missing value support --
# self-play pins the critic's support to its init and BC never fits one. `_check_warm_start` names
# this now, but naming it here costs nothing and saves the queue slot.
python - "$INIT" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
missing = [k for k in ("v_min", "v_max", "n_bins") if ck.get(k) is None]
if missing:
    raise SystemExit(f"[job] FATAL: {sys.argv[1]} has no value support ({', '.join(missing)}); "
                     f"warm-start from the offline-RL checkpoint, not the BC one.")
print(f"[job] warm start ok: {ck.get('battle_format')} / {ck.get('obs_version')} / "
      f"{ck.get('input_dim')}d / {ck.get('n_actions')} actions")
PY

ARGS=(--init "$INIT" --arm "$ARM" --format "$FMT" --seeds "$SEEDS" --iters "$ITERS"
      --games-per-opp "$GAMES_PER_OPP" --eval-n "$EVAL_N" --ladder-n "$LADDER_N"
      --out "$OUT" --device cpu)
[[ "$FMT" == *random* ]] || ARGS+=(--team-pool "$TEAM_POOL")
[[ -z "$MAX_BATTLE_TURNS" ]] || ARGS+=(--max-battle-turns "$MAX_BATTLE_TURNS")

start_showdown

echo "[job] selfplay gate $ARM on $FMT: ${ARGS[*]}"
python -u scripts/curriculum_gate.py "${ARGS[@]}"

echo "[job] $ARM done -> $OUT"

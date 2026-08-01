#!/usr/bin/env bash
# The AUTHORITATIVE build-vs-build strength gate (scripts/seed_strength_gate.py).
#
# Scores EVERY seed's best checkpoint at --n and pools to 900 battles/arm. Never hand it a
# single-seed gate file: it will pool 300 instead, still print a verdict, and have silently lost
# the power the protocol exists to buy. Merge first with scripts/merge_gate_seeds.py.
#
# One build (gets that build's pooled rate + CI, no comparison):
#   BUILD_A=v22 GATE_A=results/ppo_ou_gate_v22.json OUT=results/seed_strength_gate_v22.json \
#     sbatch -p tron --qos=medium scripts/cluster/strength_gate.slurm
#
# Two builds (the real comparison -- both arms scored in the SAME run, same server, same day,
# which is what keeps the DIFFERENCE trustworthy; absolute rates carry a winner's-curse bias
# that only cancels because the protocol is identical across arms):
#   BUILD_A=v21 GATE_A=results/ppo_ou_gate_v21.json \
#   BUILD_B=v22 GATE_B=results/ppo_ou_gate_v22.json \
#   OUT=results/seed_strength_gate_v22.json sbatch -p tron --qos=medium scripts/cluster/strength_gate.slurm
#
# Each arm's best checkpoints must be ON DISK -- /checkpoints/ is gitignored, so a build trained
# on another machine needs its three best_checkpoint files staged first. The preflight below names
# exactly which, rather than failing an hour into battles.
#
#SBATCH --job-name=lg-strength
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
# NOTE: no --gres=gpu. NOTE: needs -p tron --qos=medium (default QoS caps CPUs at 4).

set -euo pipefail
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$COMMON_DIR/_job_common.sh" ]] || COMMON_DIR="${SLURM_SUBMIT_DIR:-$COMMON_DIR}/scripts/cluster"
source "$COMMON_DIR/_job_common.sh"

BUILD_A="${BUILD_A:?set BUILD_A, e.g. v22}"
GATE_A="${GATE_A:?set GATE_A, the MERGED multi-seed gate json}"
OUT="${OUT:?set OUT, the result json path}"
N="${N:-300}"
OPPONENT="${OPPONENT:-heuristic}"
LOOP_PENALTY="${LOOP_PENALTY:-4}"

ARGS=(--build "$BUILD_A" "$GATE_A")
[[ -n "${BUILD_B:-}" ]] && ARGS+=(--build "$BUILD_B" "${GATE_B:?set GATE_B when BUILD_B is set}")

# Preflight: every arm's per-seed best_checkpoint must exist. A missing one is the gitignored
# /checkpoints/ problem, and it must fail NOW, not after the first arm's 900 battles.
python - "$GATE_A" "${GATE_B:-}" <<'PY'
import json
import sys
from pathlib import Path

missing = []
for gate in [p for p in sys.argv[1:] if p]:
    d = json.loads(Path(gate).read_text())
    n = len(d["records"])
    if n != 3:
        print(f"[preflight] WARNING: {gate} holds {n} seed record(s), not 3 -- the gate will "
              f"pool {n * 300} not 900.", file=sys.stderr)
    for r in d["records"]:
        ck = Path(r["best_checkpoint"])
        status = "ok" if ck.exists() else "MISSING"
        print(f"[preflight] {gate} seed {r['seed']}: {ck} [{status}]", file=sys.stderr)
        if not ck.exists():
            missing.append(str(ck))

if missing:
    print(
        "[preflight] FATAL: best checkpoints absent from this node. /checkpoints/ is gitignored,\n"
        "            so a build trained elsewhere must have these staged:\n"
        + "".join(f"              {m}\n" for m in missing)
        + "            e.g. rsync -avR <laptop>:<repo>/./" + missing[0] + " $REPO_DIR/",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

start_showdown

echo "[job] scoring: ${ARGS[*]}  (n=$N per checkpoint, opponent=$OPPONENT)"
python -u scripts/seed_strength_gate.py \
  "${ARGS[@]}" \
  --opponent      "$OPPONENT" \
  --n             "$N" \
  --loop-penalty  "$LOOP_PENALTY" \
  --out           "$OUT"

echo "[job] strength gate done -> $OUT"

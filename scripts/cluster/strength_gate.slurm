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
# Build 24: up to FIVE arms (BUILD_A..BUILD_E), every pair compared. Pass the pre-registered
# Bonferroni ALPHA -- with k arms there are k(k-1)/2 tests and the 0.05 default is only right for
# two. Raising N is the cheapest power available, and Build 24 needs it: its half-effects are
# ~0.030, and at alpha=0.0125 the MDE at 80% power is 0.079 / 0.045 / 0.032 for N=300 / 900 / 1800.
# Below N=1800 the likely outcome is "total significant, both halves NULL", which answers nothing.
# 27,000 battles is ~2.3h against the 8h limit -- battles are the CHEAPEST power in a build whose
# training is 15-18h. (This buys power for the POOLED test only; the seed-level SE barely moves,
# because that variance is BETWEEN seeds. See seed_strength_gate.py's two-inference-levels note.)
#
#   BUILD_A=v23b GATE_A=results/ppo_ou_gate_v23b.json \
#   BUILD_B=v24a GATE_B=results/ppo_ou_gate_v24a.json \
#   BUILD_C=v24b GATE_C=results/ppo_ou_gate_v24b.json \
#   BUILD_D=v24c GATE_D=results/ppo_ou_gate_v24c.json \
#   BUILD_E=v24d GATE_E=results/ppo_ou_gate_v24d.json \
#   N=1800 ALPHA=0.0125 OUT=results/seed_strength_gate_v24.json \
#     sbatch -p tron --qos=medium scripts/cluster/strength_gate.slurm
#
# Each arm's best checkpoints must be ON DISK -- /checkpoints/ is gitignored, so a build trained
# on another machine needs its three best_checkpoint files staged first. The preflight below names
# exactly which, rather than failing an hour into battles.
#
#SBATCH --job-name=ra-strength
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
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
# With >2 arms the pairwise tests are MULTIPLE COMPARISONS. Pass the PRE-REGISTERED Bonferroni
# level (Build 24: 4 planned contrasts -> 0.0125); the 0.05 default is only right for two arms.
ALPHA="${ALPHA:-0.05}"

# Build 24 scores a 2x2 factorial plus a decomposition arm, so arms are collected A..E rather
# than the old BUILD_A/BUILD_B pair. Every arm must be in ONE invocation -- see the calibration
# finding in the header.
ARGS=()
GATES=()
for L in A B C D E; do
  bvar="BUILD_$L"
  gvar="GATE_$L"
  bval="${!bvar:-}"
  if [[ -n "$bval" ]]; then
    gval="${!gvar:?set $gvar when $bvar is set}"
    ARGS+=(--build "$bval" "$gval")
    GATES+=("$gval")
  fi
done

# Preflight: every arm's per-seed best_checkpoint must exist. A missing one is the gitignored
# /checkpoints/ problem, and it must fail NOW, not after the first arm's 900 battles.
python - "${GATES[@]}" <<'PY'
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
  --alpha         "$ALPHA" \
  --out           "$OUT"

echo "[job] strength gate done -> $OUT"

# Optional SECOND opponent, scored in this same allocation -- same server, same node, same hour.
# Two sbatch submissions would answer a different question: seed_strength_gate.py:29-34 records
# identical checkpoints scoring 0.472 and 0.499 in two separate runs, so a control read taken on
# another day inherits exactly the variance it exists to rule out. Unset changes nothing.
if [[ -n "${OPPONENT_2:-}" ]]; then
  OUT_2="${OUT_2:?set OUT_2 when OPPONENT_2 is set}"
  echo "[job] control read: opponent=$OPPONENT_2 (n=$N per checkpoint)"
  python -u scripts/seed_strength_gate.py \
    "${ARGS[@]}" \
    --opponent      "$OPPONENT_2" \
    --n             "$N" \
    --loop-penalty  "$LOOP_PENALTY" \
    --alpha         "$ALPHA" \
    --out           "$OUT_2"
  echo "[job] control read done -> $OUT_2"
fi

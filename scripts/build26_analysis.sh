#!/usr/bin/env bash
# Build 26's analysis, pre-staged: merge -> strength gate -> pinned-terminal co-primary read.
#
# WHY THIS IS A SCRIPT AND NOT A CHECKLIST. Build 25's analysis was a hand-run chain, and two of
# its steps are silent-failure shaped: `merge_gate_seeds` will happily merge two seeds instead of
# three (halving the gate's power while still printing a verdict), and `seed_strength_gate` will
# read its default alpha=0.05 on a k>2 factorial run unless the pre-registered Bonferroni level is
# passed explicitly. Both are encoded here so the verdict cannot be produced the wrong way.
#
# The bias table is NOT applied by this script -- it is printed alongside, because plan.md's
# pre-registration requires the subtraction to be reported as an explicit, visible step:
# a contrast must clear BOTH p < alpha AND diff - bias > 0.
#
# Usage:
#   bash scripts/build26_analysis.sh --dry-run     # check inputs, print the plan, run nothing
#   bash scripts/build26_analysis.sh               # run it
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ---- Pre-registered constants (plan.md 13, Build 26). Do not edit to fit the data. ----
ANCHOR="v25b";  ANCHOR_ITERS=160
ARM_A="v26a";   ARM_A_ITERS=240
ARM_B="v26b";   ARM_B_ITERS=320
SEEDS=(0 1 2)
# 3 pre-registered contrasts (160->240, 240->320, 160->320) => Bonferroni 0.05/3.
ALPHA=0.0167
BIAS_JSON="results/selection_bias_v26.json"
LADDER_SRC="results/ppo_ou_gate_v25b.json"

say() { printf '\n=== %s ===\n' "$*"; }
run() {
  printf '  $ %s\n' "$*"
  [[ $DRY_RUN -eq 1 ]] || "$@"
}

# ---- Preflight: refuse an incomplete arm set rather than silently pooling fewer seeds. ----
#
# EXISTENCE IS NOT ENOUGH, AND THIS BUILD IS THE FIRST WHERE IT NEVER WAS. `v26b` runs in two
# chunks (160, then --resume to 320) because arm length is bounded by MEMORY, and EACH CHUNK
# WRITES THE SAME results/ppo_ou_gate_v26b_s<N>.json. So between the chunks the file is present,
# well-formed, and 160 iterations long -- and an existence-only check passes on it.
#
# What that costs, concretely: contrast #2 would compare 240 -> 160 instead of 240 -> 320 (i.e.
# backwards, reading as a large REVERSAL) and #3 would compare 160 -> 160 (a near-null). Against
# the pre-registered table that is the "sig < 0" row: a fabricated REVERSAL verdict, printed with
# every appearance of having passed its checks. Observed live on 2026-08-11 09:32, with chunk 2 at
# iteration ~290/320 and all six files sitting on disk marked ok.
#
# So each arm is also checked for LENGTH: the curve must actually reach the pre-registered
# iteration count. The seed-count guard below was written before arm length was a variable.
check_arm() {  # arm, expected_iters
  local arm="$1" want="$2" f s got
  for s in "${SEEDS[@]}"; do
    f="results/ppo_ou_gate_${arm}_s${s}.json"
    if [[ ! -f "$f" ]]; then
      echo "  MISSING  $f"
      missing=$((missing + 1))
      continue
    fi
    got="$(python - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
recs = d.get("records") or [d.get("record")]
curve = (recs[0] or {}).get("curve") or []
print(max((p["iter"] for p in curve), default=-1))
PY
)"
    if [[ "$got" == "$want" ]]; then
      echo "  ok       $f  (curve reaches iter $got)"
    else
      echo "  TOO SHORT $f  (curve reaches iter $got, pre-registered $want)"
      missing=$((missing + 1))
    fi
  done
}

say "preflight"
missing=0
check_arm "$ARM_A" "$ARM_A_ITERS"
check_arm "$ARM_B" "$ARM_B_ITERS"
[[ -f "results/ppo_ou_gate_${ANCHOR}.json" ]] \
  && echo "  ok       results/ppo_ou_gate_${ANCHOR}.json (anchor, reused)" \
  || { echo "  MISSING  results/ppo_ou_gate_${ANCHOR}.json"; missing=$((missing + 1)); }

if [[ $missing -gt 0 ]]; then
  echo
  echo "REFUSING: $missing input(s) missing or short. A partial merge halves the gate's power, and"
  echo "a SHORT ARM silently re-points a contrast at the wrong iteration count -- both still print"
  echo "a verdict, which is exactly the failure this preflight exists to prevent."
  echo "Check squeue -- Build 26 is 6 tasks (v26a s0-2, v26b s0-2), and v26b takes TWO submissions"
  echo "(160, then RESUME=1 to 320) that write the same per-seed JSON."
  [[ $DRY_RUN -eq 1 ]] || exit 1
fi

# ---- 1. Merge each arm's seeds. ----
say "1. merge seeds -> one gate JSON per arm"
for spec in "$ARM_A:$ARM_A_ITERS" "$ARM_B:$ARM_B_ITERS"; do
  arm="${spec%%:*}"; iters="${spec##*:}"
  args=()
  for s in "${SEEDS[@]}"; do args+=(--seed-json "results/ppo_ou_gate_${arm}_s${s}.json"); done
  run python scripts/merge_gate_seeds.py "${args[@]}" \
    --ladder-source "$LADDER_SRC" \
    --note "Build 26 arm ${arm}: update-count dose (target_kl 0.06 / kl_bar 0.09, iters ${iters}, anneal horizon PINNED to 80). Extends Build 25's STILL CLIMBING verdict; scored against ${ANCHOR} at alpha=${ALPHA} over 3 pre-registered contrasts." \
    --out "results/ppo_ou_gate_${arm}.json"
done

# ---- 2. Primary read: seed-best, all three arms in ONE invocation. ----
say "2. PRIMARY (seed-best) strength gate, alpha=${ALPHA}"
run python scripts/seed_strength_gate.py \
  --build "$ANCHOR" "results/ppo_ou_gate_${ANCHOR}.json" \
  --build "$ARM_A"  "results/ppo_ou_gate_${ARM_A}.json" \
  --build "$ARM_B"  "results/ppo_ou_gate_${ARM_B}.json" \
  --alpha "$ALPHA" \
  --out results/seed_strength_gate_v26.json

# ---- 3. Co-primary read: pin every arm to its TERMINAL iteration (zero selection bias). ----
# Build 25 promoted this from descriptive to co-primary because its contrast #2 was
# pooled-significant but NOT seed-robust on the seed-best read, while the terminal read was the
# stronger one (t = +2.34, 3/3) -- the reverse of the usual direction.
say "3. CO-PRIMARY (pinned terminal) strength gate"
for spec in "$ANCHOR:$ANCHOR_ITERS" "$ARM_A:$ARM_A_ITERS" "$ARM_B:$ARM_B_ITERS"; do
  arm="${spec%%:*}"; iters="${spec##*:}"
  run python scripts/pin_gate_checkpoint.py \
    --gate "results/ppo_ou_gate_${arm}.json" \
    --iter "$iters" \
    --out "results/ppo_ou_gate_${arm}_terminal.json"
done
run python scripts/seed_strength_gate.py \
  --build "$ANCHOR" "results/ppo_ou_gate_${ANCHOR}_terminal.json" \
  --build "$ARM_A"  "results/ppo_ou_gate_${ARM_A}_terminal.json" \
  --build "$ARM_B"  "results/ppo_ou_gate_${ARM_B}_terminal.json" \
  --alpha "$ALPHA" \
  --out results/seed_strength_gate_v26_terminal.json

# ---- 4. The bias subtraction, reported explicitly. ----
say "4. pre-registered differential selection bias (sigma_b = 0.0328)"
if [[ -f "$BIAS_JSON" ]]; then
  python - "$BIAS_JSON" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
entry = next(e for e in data["sweep"] if abs(e["sigma_b"] - 0.0328) < 1e-9)
print("  | contrast        | bias    |")
print("  |-----------------|---------|")
for c in entry["contrasts"]:
    print(f"  | {c['contrast']:<15} | {c['differential_bias']:+.4f} |")
print()
print("  A contrast counts only if it clears BOTH p < alpha AND (diff - bias) > 0.")
print("  Report the correction as APPLIED even when it does not bind (Build 25 precedent).")
PY
else
  echo "  MISSING $BIAS_JSON -- regenerate with scripts/selection_bias_sim.py before reporting."
fi

say "done"
echo "  primary:    results/seed_strength_gate_v26.json"
echo "  co-primary: results/seed_strength_gate_v26_terminal.json"
echo
echo "  Both reads must agree in SIGN for a contrast to be called. A disagreement is itself the"
echo "  finding and is booked as one -- never resolved toward whichever read fits the hypothesis."
echo "  Also run scripts/ppo_telemetry.py per seed for the trust-region certificate: without it a"
echo "  NULL cannot be attributed to update count."
[[ $DRY_RUN -eq 1 ]] && echo && echo "(dry run -- nothing above was executed)"
exit 0

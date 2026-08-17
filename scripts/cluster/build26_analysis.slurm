#!/usr/bin/env bash
# Build 26's whole analysis as ONE submission: server + merge + both strength gates + bias table.
#
#   sbatch -p tron --qos=medium scripts/cluster/build26_analysis.slurm
#
# Smoke the plumbing cheaply first if you want (minutes, and the verdict is MEANINGLESS at this N
# -- see the power note in build26_analysis.sh):
#   N_PRIMARY=50 N_TERMINAL=50 sbatch -p tron --qos=medium scripts/cluster/build26_analysis.slurm
#
# WHY A WRAPPER EXISTS AT ALL. `build26_analysis.sh` calls `seed_strength_gate.py` directly, and
# that plays real battles -- so it needs a Showdown server, which nothing in that script starts.
# At the pre-registered N it is ~43,200 battles, roughly 3 h at the 242 battles/min Build 25
# measured. That is not a login-node job, and it is long enough that "I'll start the server by
# hand" is exactly the step that gets forgotten.
#
# THE PORT BASE MATTERS WHILE A PPO BUILD IS IN FLIGHT. _job_common.sh computes 8100 + TASK_ID and
# a non-array job takes TASK_ID=0 -> 8100, which is the port `ppo_seed.slurm`'s array task 0 claims.
# Colliding jobs on one node SILENTLY SHARE a server rather than failing, so they see each other's
# battles. 8400 is clear of both PPO bases (8100-8102, 8200-8202) and of eval_ladder.slurm (8300).
#
#SBATCH --job-name=ra-b26-analysis
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/slurm-%x-%j.out
# NOTE: no --gres=gpu. NOTE: needs -p tron --qos=medium (default QoS caps CPUs at 4).
#
# Build 26's own six training tasks sit exactly at the tron QoS's cpu=32 per-user cap, so this will
# stay PENDING with reason QOSMaxCpuPerUserLimit until they finish. That is the correct order
# anyway -- the preflight refuses a short arm.

set -euo pipefail
export ROTOMAI_SHOWDOWN_PORT_BASE="${ROTOMAI_SHOWDOWN_PORT_BASE:-8400}"

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$COMMON_DIR/_job_common.sh" ]] || COMMON_DIR="${SLURM_SUBMIT_DIR:-$COMMON_DIR}/scripts/cluster"
source "$COMMON_DIR/_job_common.sh"

# Fail on a short or missing arm BEFORE burning a server and a node on it. This runs the SAME
# preflight the real invocation does -- captured once, so what is printed and what is judged cannot
# be two different things.
echo "[job] preflight (dry run) -- refuses a short arm, e.g. v26b still mid-chunk-2"
PREFLIGHT="$(bash scripts/build26_analysis.sh --dry-run)"
sed -n '/=== preflight ===/,/^$/p' <<<"$PREFLIGHT"
if grep -qE "MISSING|TOO SHORT" <<<"$PREFLIGHT"; then
  echo "[job] FATAL: an arm is missing or short -- see above. Not starting a server." >&2
  exit 1
fi

start_showdown

echo "[job] running the analysis (N_PRIMARY=${N_PRIMARY:-3000}, N_TERMINAL=${N_TERMINAL:-1800})"
bash scripts/build26_analysis.sh

echo "[job] Build 26 analysis done"
echo "[job]   primary:    results/seed_strength_gate_v26.json"
echo "[job]   co-primary: results/seed_strength_gate_v26_terminal.json"

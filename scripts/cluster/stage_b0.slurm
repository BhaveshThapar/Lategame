#!/usr/bin/env bash
# Build 22 STAGE B-0 -- the cheap qualifier (docs/RESULTS.md, pre-registered 2026-07-20).
#
# Stage A confirmed H22: the wide net is BORN FLAT (|G|^2(0) = 0.064 ~ its Build-21 iter_10 value
# 0.057; cos 0.186, BELOW the 0.3 floor). B-0 asks whether the OFFLINE FIT is what flattens it, by
# training the SAME wide arch on the SAME shard for FEWER epochs and re-probing |G|^2 at iter_0.
#
#   weakened |G|^2(0) >> 0.064 AND cos > 0.3  => offline fit flattens it; mechanism ACTIONABLE
#                                                -> proceed to Stage B (3-seed PPO)
#   weakened |G|^2(0) ~  0.064,    cos < 0.3  => offline convergence is NOT the flattener
#                                                -> CANCEL PPO; the flat point is architectural
#
# Read the WITHIN-architecture trend. Every arm here is the SAME width, so |G|^2 IS comparable
# across them -- unlike Stage A's wide-vs-narrow, where only cos and B_simple were.
#
# IGNORE the script's own NOISE_LIMITED / AMBIGUOUS verdict. That is grad_noise_diag's Build-20
# sample-budget logic; B_simple is the ratio tr(Sigma)/|G|^2 and EXPLODES when |G|^2 -> 0. "Collect
# more samples" is exactly wrong against a vanished gradient.
#
#SBATCH --job-name=lg-b22-stageB0
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/slurm-%x-%A_%a.out
# NOTE: no --gres=gpu. The net is 4.56M params; a GPU buys nothing and lengthens the queue.
# NOTE: --partition is intentionally unset -- fill in whatever UMIACS gives you. Stage A ran on
#       `tron` (sacct 7114674): wide probe 57 min, narrow 13 min.
# NOTE: 8h, not 4h. Each task is TRAIN + PROBE, and the probe alone measured ~57 min for this
#       width. The 30-epoch arm's train time is UNMEASURED (the reference net was trained on the
#       laptop), and grad_noise_diag serializes ONLY AT THE END -- a wall-clock kill loses the
#       whole task, the exact failure this project already ate once.

set -euo pipefail
# sbatch runs a spool COPY of this script, so ${BASH_SOURCE[0]}'s sibling _job_common.sh is NOT
# beside it at run time. Try the sibling first (direct `bash` runs), else fall back to the submit
# dir -- the repo root, per scripts/cluster/README.md's documented `sbatch` cwd.
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$COMMON_DIR/_job_common.sh" ]] || COMMON_DIR="${SLURM_SUBMIT_DIR:-$COMMON_DIR}/scripts/cluster"
source "$COMMON_DIR/_job_common.sh"

# The RL shard that built offrl_gen9ou_wide_s0.pt. NOT in git (/data/ is gitignored) and NOT on
# this node -- stage it before submitting. Do NOT regenerate it with `fetch-replays`: that walks
# Showdown's LIVE search index, so a re-fetch returns a DIFFERENT replay set and the dose-response
# becomes confounded by data instead of by epochs. Copy the original.
SHARD="${SHARD:-data/gen9ou_v7_rl.npz}"
# The converged (default 30-epoch) wide net. Read as the arch + shard reference, never overwritten.
REF_CKPT="${REF_CKPT:-checkpoints/offrl_gen9ou_wide_s0.pt}"

# Array task -> offline dose. 0 and 1 are the pre-registered weakened arms.
#
# Task 2 (epochs 30 = the trainer default) is an ADDITION, not a change to the gate: it reproduces
# the converged wide net ON THIS NODE. The 0.064 reference was trained on the laptop, so without it
# a difference between arms could be device numerics rather than epochs. It also re-validates the
# whole pipeline -- a 30-epoch retrain here should land back at |G|^2(0) ~ 0.064. Drop it with
# `--array=0-1` to run the pre-registration literally.
case "$SLURM_ARRAY_TASK_ID" in
  0) EPOCHS=3  ;;
  1) EPOCHS=10 ;;
  2) EPOCHS=30 ;;
  *) echo "unexpected array index" >&2; exit 1 ;;
esac

CKPT="checkpoints/offrl_gen9ou_wide_e${EPOCHS}_s0.pt"
OUT="results/grad_noise_diag_b22_stageB0_e${EPOCHS}.json"

# ---------------------------------------------------------------------------
# PREFLIGHT -- fails in seconds, before an hour of training is spent on the wrong inputs.
#
# Two things must hold, and neither is checkable by eye:
#   1. The arch must be READ from the reference checkpoint, not guessed (pre-registration is
#      explicit about this). Emitted as shell vars below.
#   2. The shard must be THE SAME ONE. train_offline_rl derives the value support from the
#      shard's own returns (_support_from_returns, offline_rl.py:115) and stamps it into the
#      checkpoint -- so the reference's v_min/v_max are a FINGERPRINT of its training shard.
#      A different replay set moves them well beyond this tolerance.
# RLDataset additionally raises on an obs_version/dim mismatch (rl_dataset.py:39-44), so a
# stale-encoder shard cannot slip through either.
# ---------------------------------------------------------------------------
echo "[job] preflight: shard=$SHARD ref=$REF_CKPT"
if [[ ! -f "$SHARD" ]]; then
  echo "[job] FATAL: RL shard not found at $SHARD" >&2
  echo "[job]        Stage it from the machine that trained the wide net, e.g." >&2
  echo "[job]          rsync -avP <laptop>:<repo>/data/gen9ou_v7_rl.npz $REPO_DIR/data/" >&2
  echo "[job]        Do NOT re-run fetch-replays: a fresh fetch is a different shard." >&2
  exit 1
fi
[[ -f "$REF_CKPT" ]] || { echo "[job] FATAL: reference checkpoint not found at $REF_CKPT" >&2; exit 1; }

PREFLIGHT="$(python - "$SHARD" "$REF_CKPT" <<'PY'
import sys

import torch

from lategame.data.rl_dataset import RLDataset
from lategame.train.offline_rl import _support_from_returns

shard_path, ref_path = sys.argv[1], sys.argv[2]
ref = torch.load(ref_path, map_location="cpu", weights_only=False)

# Raises on an encoder-version mismatch.
dataset = RLDataset(shard_path)
v_min, v_max = _support_from_returns(dataset.ret)
r_min, r_max = float(ref["v_min"]), float(ref["v_max"])

print(f"[preflight] shard rows      : {len(dataset)}", file=sys.stderr)
print(f"[preflight] shard format    : {dataset.battle_format}", file=sys.stderr)
print(f"[preflight] support (shard)  : [{v_min:.9f}, {v_max:.9f}]", file=sys.stderr)
print(f"[preflight] support (ref ckpt): [{r_min:.9f}, {r_max:.9f}]", file=sys.stderr)

if abs(v_min - r_min) > 1e-6 or abs(v_max - r_max) > 1e-6:
    print(
        "[preflight] FATAL: this shard's value support does not match the one stamped into\n"
        f"            {ref_path}. The shard is NOT the one that trained the reference net, so a\n"
        "            B-0 dose-response run on it would confound offline epochs with a change of\n"
        "            data. Stage the original shard.",
        file=sys.stderr,
    )
    raise SystemExit(1)

if ref.get("battle_format") != dataset.battle_format:
    print(
        f"[preflight] FATAL: format mismatch -- shard {dataset.battle_format!r} vs reference "
        f"{ref.get('battle_format')!r}.",
        file=sys.stderr,
    )
    raise SystemExit(1)

arch = ref["arch"]
for key in ("d_model", "n_layers", "n_heads", "ff_dim"):
    print(f"ARCH_{key.upper()}={int(arch[key])}")
print(f"ARCH_ID_EMBED_INIT={arch.get('id_embed_init', 'prior')}")
print(f"REF_N_BINS={int(ref['n_bins'])}")
PY
)"
eval "$PREFLIGHT"
echo "[job] preflight OK -- arch d_model=$ARCH_D_MODEL n_layers=$ARCH_N_LAYERS" \
     "n_heads=$ARCH_N_HEADS ff_dim=$ARCH_FF_DIM n_bins=$REF_N_BINS (read from $REF_CKPT)"

# ---------------------------------------------------------------------------
# 1. Retrain the wide net at a capped offline dose.
#
# train_offline_rl saves BEST-BY-VAL-LOSS (offline_rl.py:274-278), so --epochs caps how converged
# the saved net can be -- that IS the dose. --bc-init '' trains from scratch: the transformer has
# no compatible MLP trunk, and Build 21's control showed the BC warm start contributes NOTHING
# (0.6485 accuracy either way). (_run_train_rl already drops the default BC init for non-MLP
# models; passing it explicitly means a future default change cannot silently warm-start this.)
# No Showdown needed -- this reads an npz off disk.
# ---------------------------------------------------------------------------
echo "[job] === training wide @ ${EPOCHS} epochs -> $CKPT ==="
python -u -m lategame.cli train-rl \
  --data           "$SHARD" \
  --out            "$CKPT" \
  --bc-init        '' \
  --model          entity_transformer \
  --d-model        "$ARCH_D_MODEL" \
  --n-layers       "$ARCH_N_LAYERS" \
  --n-heads        "$ARCH_N_HEADS" \
  --ff-dim         "$ARCH_FF_DIM" \
  --id-embed-init  "$ARCH_ID_EMBED_INIT" \
  --n-bins         "$REF_N_BINS" \
  --epochs         "$EPOCHS" \
  --seed           0

# ---------------------------------------------------------------------------
# 2. Probe |G|^2 at iter_0 -- the warm start itself, before a single PPO step.
#
# Protocol is IDENTICAL to Stage A, which is what makes 0.064 a valid reference: gpo 48 is the
# budget both v20 and v21 actually trained at, and --splits DEFAULTS TO 20, so passing 5 is
# mandatory for comparability. --policy == --init probes the warm start; _league_for handles k=0
# (the league is [init] + anchors, exactly the mix iteration 0 faced).
# ---------------------------------------------------------------------------
start_showdown

echo "[job] === probing |G|^2 at iter_0 for $CKPT ==="
python -u scripts/grad_noise_diag.py \
  --policy        "$CKPT" \
  --init          "$CKPT" \
  --games-per-opp 48 \
  --rollouts      6 \
  --splits        5 \
  --out           "$OUT"

echo "[job] Stage B-0 (epochs=${EPOCHS}) done -> $OUT"
echo "[job] compare same_mix.policy.noise_scale.g_norm_sq against the converged wide's 0.064,"
echo "[job] and cos@budget against 0.186 / the 0.3 floor."

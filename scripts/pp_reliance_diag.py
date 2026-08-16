"""pp-reliance diagnostic (docs/RESULTS.md, Build 11) -- promoted from the Build-10 inline snippet.

Measures how much a policy's live switch mass is carried by the *exactly-full* pp cue, on a
frozen set of behavior-probe states. For each present move-pp channel we replace the live value
with an in-distribution draw from the training shard's pp pool (frac_full ~= 0.50) and re-score:

    dP = baseline_switch_mass - pp_neutralized_switch_mass

A large ``dP`` means the policy keys on exactly-full pp (the loop carrier, Build 8). Build 11's
question is whether a global pp regularizer (``augment_pp_noise`` / ``augment_pp_resample``) pushes
``dP`` materially below the v10 (0.390) / v7 (0.397) reference on the SAME frozen v7 states.

Row selection is the FULL capture (the pathological agent's whole behavior probe is loop behavior);
its frac_full matches the Build-10 report (~0.861). Scoring reuses ``switch_mass_gate.load_policy``
+ ``score_shard`` so switch mass is defined identically to every other gate.

Example:
    python scripts/pp_reliance_diag.py \
      --policy checkpoints/bc_gen9ou_v11_s0.pt \
      --states results/behavior_probe_obs_v7.npz \
      --shard data/gen9ou_v7_bc.npz \
      --out results/pp_reliance_diag11.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import switch_mass_gate as smg  # noqa: E402

from lategame.features.encoder import OBS_DIM, OBS_LAYOUT, OBS_VERSION  # noqa: E402

_MOVE_PP_IDX = 4
_MOVE_PRESENT_IDX = 0
_FULL = 0.999


def _pp_present_cols() -> tuple[list[int], list[int]]:
    """Flat-obs indices of the active mon's (pp, present) channels, one per move."""
    ms, md, n = OBS_LAYOUT.moves_start, OBS_LAYOUT.move_dim, OBS_LAYOUT.n_moves
    pp = [ms + j * md + _MOVE_PP_IDX for j in range(n)]
    present = [ms + j * md + _MOVE_PRESENT_IDX for j in range(n)]
    return pp, present


def shard_pp_pool(shard_path: str) -> np.ndarray:
    """Present-move pp values in the training shard -- the in-distribution neutralization pool."""
    npz = np.load(shard_path)
    obs = np.asarray(npz["obs"], dtype=np.float32)
    pp_cols, present_cols = _pp_present_cols()
    pp = obs[:, pp_cols]
    present = obs[:, present_cols] > 0.0
    return pp[present]


def frac_full(obs: np.ndarray) -> float:
    """Fraction of present-move pp cells sitting at (near) full pp."""
    pp_cols, present_cols = _pp_present_cols()
    pp = obs[:, pp_cols]
    present = obs[:, present_cols] > 0.0
    vals = pp[present]
    return float((vals >= _FULL).mean()) if vals.size else float("nan")


def neutralize_pp(obs: np.ndarray, pool: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of ``obs`` with every present-move pp channel resampled from ``pool``."""
    out = obs.copy()
    pp_cols, present_cols = _pp_present_cols()
    for pp_col, present_col in zip(pp_cols, present_cols, strict=True):
        rows = out[:, present_col] > 0.0
        n = int(rows.sum())
        if n:
            out[rows, pp_col] = pool[rng.integers(pool.size, size=n)]
    return out


def run(
    policy_path: str,
    states_path: str,
    shard_path: str,
    *,
    batch_size: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    states = np.load(states_path)
    obs = np.asarray(states["obs"], dtype=np.float32)
    mask = np.asarray(states["mask"])

    model = smg.load_policy(policy_path, OBS_VERSION, OBS_DIM)
    baseline, _ = smg.score_shard(model, obs, mask, batch_size, device)

    pool = shard_pp_pool(shard_path)
    obs_neut = neutralize_pp(obs, pool, np.random.default_rng(seed))
    neutralized, _ = smg.score_shard(model, obs_neut, mask, batch_size, device)

    return {
        "policy": policy_path,
        "states": states_path,
        "shard": shard_path,
        "n_rows": int(obs.shape[0]),
        "shard_pp_pool_frac_full": round(float((pool >= _FULL).mean()), 4),
        "frac_full_live": round(frac_full(obs), 4),
        "baseline_switch_mass": round(float(baseline.mean()), 4),
        "pp_neutralized_switch_mass": round(float(neutralized.mean()), 4),
        "dP": round(float(baseline.mean() - neutralized.mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True, help="BC checkpoint to score")
    ap.add_argument("--states", default="results/behavior_probe_obs_v7.npz", help="frozen states")
    ap.add_argument("--shard", default="data/gen9ou_v7_bc.npz", help="shard for the pp pool")
    ap.add_argument("--out", default="results/pp_reliance_diag.json")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    result = run(
        args.policy,
        args.states,
        args.shard,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    result["note"] = "dP = baseline - pp_neutralized. Large dP => policy keys on exactly-full pp."
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

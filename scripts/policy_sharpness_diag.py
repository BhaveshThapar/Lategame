"""Build 19 (docs/RESULTS.md): is the PPO plateau an OPTIMIZATION ceiling?

Build 18 plateaued PPO self-play at ``vs_heuristic`` ~0.45 (n=300). The pre-registered
suspect is the *fixed* ``ent_coef=0.01`` / ``lr=2.5e-4`` schedule (ppo.py) -- but the naive
"the policy is too random to close out wins" story is already falsified by the code:

    rollout SAMPLES  (ppo.py: PlayerSpec(..., sample=True); PPORecordingAgent)
    eval   is GREEDY (ppo.py: _eval_point builds the learner with sample=False -> argmax)

so residual entropy cannot *directly* cost eval win-rate. It can only hurt indirectly, by
holding the learned distribution soft so the argmax lags the distribution PPO optimizes.
That is a real train/eval objective mismatch, but it is not free -- so qualify it before
spending a 3-hour run on it.

Two measurements on the SHIPPED v18 checkpoints (no training, no code change to lategame):

  A1 sharpness  -- mean entropy / max-prob of the masked policy over frozen live decision
                   states, for the warm-start, iter_01 and the v18 best iter. Reports the
                   uniform-over-legal entropy as the soft ceiling, so "soft" is measured
                   against the actual legal-set size rather than an absolute nat count.
  A2 greedy-vs-sampled -- win-rate of the same checkpoint vs the eval opponent under
                   argmax (the SCORED policy) and under sampling (the TRAINED policy).

Pre-registered read (docs/RESULTS.md, Build 19):
  LIVE -> the converged policy is still soft (H_ratio >= 0.40) and/or sampled << greedy:
          the entropy bonus is holding the distribution soft. Run the schedule.
  DEAD -> the policy is already sharp (H_ratio <= 0.15) and sampled ~= greedy: annealing
          0.01 -> 0 changes a distribution that is already deterministic. Skip it.

Scoring reuses ``switch_mass_gate.load_policy`` and the same masked softmax the live agents
apply (``model.policy.masked_logits``); the win-rate A/B reuses ``arena.build_player`` /
``evaluate_built`` (``cli evaluate`` does not expose ``--sample``).

    python scripts/policy_sharpness_diag.py \
      --states results/behavior_probe_obs_v19.npz \
      --policy checkpoints/ppo_ou_x50_s0/iter_41.pt \
      --reference checkpoints/offrl_gen9ou_v7_s0.pt \
      --reference checkpoints/ppo_ou_x50_s0/iter_01.pt \
      --opponent heuristic --n 300 --loop-penalty 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import switch_mass_gate as smg

from lategame.features.encoder import OBS_DIM, OBS_VERSION

# Pre-registered thresholds (docs/RESULTS.md, Build 19). H_ratio = mean entropy / mean
# uniform-over-legal
# entropy, i.e. 1.0 = uniform over the legal set, 0.0 = deterministic.
SOFT_H_RATIO = 0.40  # >= this and the entropy bonus is still holding the policy soft
SHARP_H_RATIO = 0.15  # <= this and the policy is effectively deterministic already
SAMPLED_GAP = 0.05  # greedy - sampled below this counts as "sampled ~= greedy"


def sharpness(model: Any, obs: np.ndarray, mask: np.ndarray, batch_size: int, device: str) -> dict:
    """Mean entropy / max-prob / n_legal of the masked policy over ``obs``."""
    import torch

    from lategame.model.policy import masked_logits, policy_logits

    model.to(device)
    ents: list[np.ndarray] = []
    maxps: list[np.ndarray] = []
    legals: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, obs.shape[0], batch_size):
            ob = torch.from_numpy(obs[i : i + batch_size]).float().to(device)
            mk = torch.from_numpy(mask[i : i + batch_size]).to(device)
            logits = masked_logits(policy_logits(model(ob)), mk)
            log_probs = torch.log_softmax(logits, dim=1)
            probs = log_probs.exp()
            # Masked-out actions have prob 0; 0 * log 0 -> 0 (torch gives nan, so zero it).
            terms = torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs))
            ents.append((-terms.sum(dim=1)).float().cpu().numpy())
            maxps.append(probs.max(dim=1).values.float().cpu().numpy())
            legals.append(mk.bool().sum(dim=1).float().cpu().numpy())
    model.to("cpu")

    entropy = np.concatenate(ents).astype(np.float64)
    max_prob = np.concatenate(maxps).astype(np.float64)
    n_legal = np.concatenate(legals).astype(np.float64)
    uniform = np.log(np.maximum(n_legal, 1.0))  # entropy of uniform over the legal set
    mean_uniform = float(uniform.mean())
    mean_entropy = float(entropy.mean())
    return {
        "mean_entropy": round(mean_entropy, 4),
        "mean_max_prob": round(float(max_prob.mean()), 4),
        "mean_n_legal": round(float(n_legal.mean()), 2),
        "uniform_entropy": round(mean_uniform, 4),
        "h_ratio": round(mean_entropy / mean_uniform, 4) if mean_uniform > 0 else 0.0,
        "frac_near_deterministic": round(float((max_prob >= 0.9).mean()), 4),
    }


async def win_rates(
    policy: str,
    opponent: str,
    battle_format: str,
    team_pool: str,
    n: int,
    loop_penalty: float,
    concurrency: int,
) -> dict:
    """Win-rate of ``policy`` vs ``opponent`` under argmax and under sampling."""
    from lategame.eval.arena import build_player, evaluate_built
    from lategame.teambuilding.pool import TeamPool

    team = TeamPool.from_packed_file(team_pool) if team_pool else None
    rates: dict[str, float] = {}
    for label, sample in (("greedy", False), ("sampled", True)):
        learner = build_player(
            "offrl",
            battle_format,
            checkpoint_path=policy,
            sample=sample,
            max_concurrent_battles=concurrency,
            team=team,
            loop_penalty=loop_penalty,
        )
        opp = build_player(
            opponent, battle_format, max_concurrent_battles=concurrency, team=team
        )
        rates[label] = round(await evaluate_built(learner, opp, n), 4)
        print(f"  {label:>7} vs {opponent}: {rates[label]:.3f}  (n={n})")
    rates["greedy_minus_sampled"] = round(rates["greedy"] - rates["sampled"], 4)
    return rates


def verdict(policy_sharp: dict, rates: dict | None) -> tuple[str, str]:
    """Pre-registered LIVE / DEAD call (docs/RESULTS.md, Build 19)."""
    h = float(policy_sharp["h_ratio"])
    soft = h >= SOFT_H_RATIO
    sharp = h <= SHARP_H_RATIO
    gap = float(rates["greedy_minus_sampled"]) if rates else 0.0
    greedy_much_better = rates is not None and gap >= SAMPLED_GAP

    if soft or greedy_much_better:
        why = (
            f"h_ratio {h:.3f} >= {SOFT_H_RATIO} (policy still soft)"
            if soft
            else f"greedy - sampled {gap:+.3f} >= {SAMPLED_GAP} (argmax rides a distribution "
            "PPO never sharpened)"
        )
        return "LIVE", why
    if sharp and not greedy_much_better:
        return "DEAD", (
            f"h_ratio {h:.3f} <= {SHARP_H_RATIO} (already deterministic) and greedy - sampled "
            f"{gap:+.3f} < {SAMPLED_GAP}: annealing ent_coef changes a policy that is already "
            "sharp -- pivot to the sample-budget lever (games_per_opp)"
        )
    return "AMBIGUOUS", (
        f"h_ratio {h:.3f} sits between {SHARP_H_RATIO} and {SOFT_H_RATIO} and greedy - sampled "
        f"{gap:+.3f} < {SAMPLED_GAP}: no pre-registered call -- decide on the trend across "
        "warm-start -> iter_01 -> best"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True, help="checkpoint under test (v18 best iter)")
    ap.add_argument(
        "--reference",
        action="append",
        default=[],
        help="extra checkpoints to score for the trend (warm-start, iter_01); repeatable",
    )
    ap.add_argument("--states", default="results/behavior_probe_obs_v19.npz")
    ap.add_argument("--out", default="results/policy_sharpness_diag.json")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    # A2 (skip with --n 0 to run the cheap offline half only).
    ap.add_argument("--n", type=int, default=300, help="battles per arm for the greedy/sampled A/B")
    ap.add_argument("--opponent", default="heuristic")
    ap.add_argument("--format", default="gen9ou")
    ap.add_argument("--team-pool", default="lategame/teambuilding/data/teams_gen9ou.packed")
    ap.add_argument("--loop-penalty", type=float, default=4.0)
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()

    states = np.load(args.states)
    obs = np.asarray(states["obs"], dtype=np.float32)
    mask = np.asarray(states["mask"])
    print(f"A1 sharpness over {obs.shape[0]} frozen states from {args.states}")

    sharp: dict[str, dict] = {}
    for path in [*args.reference, args.policy]:
        model = smg.load_policy(path, OBS_VERSION, OBS_DIM)
        sharp[path] = sharpness(model, obs, mask, args.batch_size, args.device)
        s = sharp[path]
        print(
            f"  {path:<44} H={s['mean_entropy']:.3f} / {s['uniform_entropy']:.3f} uniform "
            f"(ratio {s['h_ratio']:.3f})  max_prob={s['mean_max_prob']:.3f}  "
            f"det_frac={s['frac_near_deterministic']:.3f}"
        )

    rates: dict | None = None
    if args.n > 0:
        print(f"\nA2 greedy-vs-sampled win-rate ({args.policy})")
        rates = asyncio.run(
            win_rates(
                args.policy,
                args.opponent,
                args.format,
                args.team_pool,
                args.n,
                args.loop_penalty,
                args.concurrency,
            )
        )

    call, why = verdict(sharp[args.policy], rates)
    result: dict[str, Any] = {
        "policy": args.policy,
        "states": args.states,
        "n_rows": int(obs.shape[0]),
        "sharpness": sharp,
        "win_rates": rates,
        "opponent": args.opponent,
        "n_battles": args.n,
        "loop_penalty": args.loop_penalty,
        "thresholds": {
            "soft_h_ratio": SOFT_H_RATIO,
            "sharp_h_ratio": SHARP_H_RATIO,
            "sampled_gap": SAMPLED_GAP,
        },
        "verdict": call,
        "why": why,
        "note": (
            "h_ratio = mean policy entropy / mean uniform-over-legal entropy. Eval is greedy "
            "(argmax) while rollout samples, so entropy can only hurt INDIRECTLY -- by keeping "
            "the distribution soft so the argmax lags. LIVE => run the ent/lr schedule."
        ),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nVERDICT {call}: {why}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

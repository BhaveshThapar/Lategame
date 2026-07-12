"""Build 20: the AUTHORITATIVE strength comparison between two PPO builds.

Supersedes the "score the single best checkpoint at n=300" protocol used through Build 19, which
Build 20 showed is not fit to decide the comparisons we actually make:

  UNDERPOWERED -- one checkpoint at n=300 has SE ~0.029, so a BUILD-vs-BUILD difference has
      SE ~0.041 and can only resolve gaps > ~0.08. Build 20's candidate effect was 0.074: under
      the floor. The gate could not have detected it even if it were real.
  SELECTION-BIASED -- that one checkpoint is the argmax over ~150 noisy n=100 curve evals (50
      iters x 3 seeds), then re-scored. It is maximally exposed to regression to the mean, and the
      shrinkage is not stable across builds (v20's best fell 0.600 -> 0.480 on re-scoring; v19's
      fell 0.493 -> 0.490), so the single-checkpoint number carries a build-dependent bias.

The fix, applied symmetrically to both builds: score EVERY SEED's best checkpoint (not the argmax
across seeds) and pool. That drops the across-seed selection step and triples the sample --
SE of the difference falls ~0.041 -> ~0.024, which resolves a 0.07 effect at z ~ 3.

Per-seed selection (best iter WITHIN a seed) is still a max over 50 noisy evals, so each pooled
rate keeps a winner's-curse bias. That bias is not removed -- it is made COMMON to both builds by
using an identical protocol, so it cancels in the DIFFERENCE, which is the quantity under test.
Absolute rates from this gate are therefore still optimistic; only the difference is trustworthy.

    python scripts/seed_strength_gate.py \
      --build v19 results/ppo_ou_gate_v19.json \
      --build v20 results/ppo_ou_gate_v20.json \
      --opponent heuristic --n 300 --loop-penalty 4 \
      --out results/seed_strength_gate_v20.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from format_ceiling_gate import wilson_ci  # type: ignore[import-not-found]

from lategame.eval.arena import build_player, evaluate_built
from lategame.teambuilding.pool import TeamPool


def two_proportion_z(w1: int, n1: int, w2: int, n2: int) -> dict[str, float]:
    """Pooled two-proportion z-test of ``p2 - p1``.

    Reported alongside the CIs because CI OVERLAP IS A CONSERVATIVE TEST: two 95% intervals can
    overlap while the difference is still significant at 0.05. Disjoint CIs imply significance;
    the converse does not hold.
    """
    p1, p2 = w1 / n1, w2 / n2
    pooled = (w1 + w2) / (n1 + n2)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0
    return {
        "diff": round(p2 - p1, 4),
        "z": round(z, 3),
        "p_value": round(math.erfc(abs(z) / math.sqrt(2)), 4),
        "se": round(se, 4),
    }


async def score_build(
    checkpoints: list[str],
    opponent: str,
    battle_format: str,
    team: TeamPool | None,
    n: int,
    loop_penalty: float,
    concurrency: int,
) -> dict[str, Any]:
    """Win-rate of every seed's best checkpoint vs ``opponent``, pooled over seeds."""
    per_ckpt: list[dict[str, Any]] = []
    for path in checkpoints:
        learner = build_player(
            "offrl",
            battle_format,
            checkpoint_path=path,
            sample=False,  # eval is greedy, matching the deployed policy
            max_concurrent_battles=concurrency,
            team=team,
            loop_penalty=loop_penalty,
        )
        opp = build_player(opponent, battle_format, max_concurrent_battles=concurrency, team=team)
        rate = await evaluate_built(learner, opp, n)
        wins = int(round(rate * n))
        lo, hi = wilson_ci(wins, n)
        per_ckpt.append(
            {"checkpoint": path, "wins": wins, "n": n, "rate": round(rate, 4), "ci95": [lo, hi]}
        )
        print(f"    {path:<45} {rate:.3f}  [{lo:.3f},{hi:.3f}]  ({wins}/{n})")

    wins = sum(c["wins"] for c in per_ckpt)
    total = sum(c["n"] for c in per_ckpt)
    lo, hi = wilson_ci(wins, total)
    pooled = {
        "wins": wins,
        "n": total,
        "rate": round(wins / total, 4),
        "ci95": [lo, hi],
        "per_checkpoint": per_ckpt,
    }
    print(f"  POOLED  {wins}/{total} = {wins / total:.3f}  ci95 [{lo:.3f},{hi:.3f}]")
    return pooled


def best_checkpoints(gate_json: str) -> list[str]:
    """Each seed's best checkpoint from a ppo_continue_gate result (NOT the argmax across seeds)."""
    gate = json.loads(Path(gate_json).read_text())
    return [r["best_checkpoint"] for r in gate["records"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--build",
        nargs=2,
        action="append",
        metavar=("NAME", "GATE_JSON"),
        required=True,
        help="build label + its ppo_continue_gate JSON; pass twice to compare",
    )
    ap.add_argument("--opponent", default="heuristic")
    ap.add_argument("--n", type=int, default=300, help="battles per checkpoint")
    ap.add_argument("--format", default="gen9ou")
    ap.add_argument("--team-pool", default="lategame/teambuilding/data/teams_gen9ou.packed")
    ap.add_argument("--loop-penalty", type=float, default=4.0)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--out", default="results/seed_strength_gate.json")
    args = ap.parse_args()

    team = TeamPool.from_packed_file(args.team_pool) if args.team_pool else None
    builds: dict[str, Any] = {}
    for name, gate_json in args.build:
        ckpts = best_checkpoints(gate_json)
        print(f"\n{name}: {len(ckpts)} seed-best checkpoints vs {args.opponent} (n={args.n} each)")
        builds[name] = asyncio.run(
            score_build(
                ckpts, args.opponent, args.format, team, args.n, args.loop_penalty, args.concurrency
            )
        )

    result: dict[str, Any] = {
        "gate": "seed_strength",
        "opponent": args.opponent,
        "n_per_checkpoint": args.n,
        "format": args.format,
        "loop_penalty": args.loop_penalty,
        "builds": builds,
        "note": (
            "Every seed's best checkpoint is scored and POOLED -- the across-seed argmax of the "
            "old protocol is dropped, and the sample triples. Per-seed 'best iter' is still a max "
            "over 50 noisy curve evals, so absolute rates stay optimistic; the protocol is "
            "identical across builds, so that bias cancels in the DIFFERENCE, which is what is "
            "under test. CI overlap is a CONSERVATIVE test -- read the z-test too."
        ),
    }

    names = [name for name, _ in args.build]
    if len(names) == 2:
        a, b = builds[names[0]], builds[names[1]]
        test = two_proportion_z(a["wins"], a["n"], b["wins"], b["n"])
        disjoint = b["ci95"][0] > a["ci95"][1] or a["ci95"][0] > b["ci95"][1]
        significant = test["p_value"] < 0.05
        result["comparison"] = {
            "baseline": names[0],
            "candidate": names[1],
            "disjoint_ci": disjoint,
            "significant_at_05": significant,
            **test,
            "verdict": "WIN" if significant and test["diff"] > 0 else "NULL",
        }
        c = result["comparison"]
        print(f"\n=== {names[0]} -> {names[1]} ===")
        print(
            f"  {a['rate']:.3f} [{a['ci95'][0]:.3f},{a['ci95'][1]:.3f}]  ->  "
            f"{b['rate']:.3f} [{b['ci95'][0]:.3f},{b['ci95'][1]:.3f}]"
        )
        print(
            f"  diff {c['diff']:+.3f}   z {c['z']:+.2f}   p {c['p_value']:.4f}   "
            f"disjoint_ci {disjoint}"
        )
        print(f"  VERDICT {c['verdict']}")

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

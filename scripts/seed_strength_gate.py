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

TWO OR MORE ARMS. Every pair is compared, into ``comparisons``; a two-arm run also keeps the
single ``comparison`` key the Build 20-23 result files use. Scoring k arms in ONE invocation is
not a convenience -- it is the standing measurement rule (Build 22's calibration finding: v20's
IDENTICAL checkpoints scored 0.472 and 0.499 in two separate runs, 0.027 apart against a 0.023 SE,
so a cross-run difference under ~0.03 is not a result). Pairwise tests over k > 2 arms are MULTIPLE
COMPARISONS: pre-register the contrasts that decide the build and pass --alpha corrected for them.

TWO INFERENCE LEVELS (Build 24). The pooled z-test is the pre-registered verdict, and it is a claim
about THESE CHECKPOINTS. Each contrast also carries a ``seed_level`` block -- the same contrast
paired by seed -- which is the claim about the TRAINING PROCEDURE. They routinely disagree, because
the between-seed sd (~0.0706 in Build 23) is 2.5x the within-seed binomial noise: Build 23's pooled
p = 0.0003 is a seed-level t of 2.04. Raising --n does NOT close that gap (the procedure-level SE
of a difference moves 0.0622 -> 0.0579 going from n=300 to n=5000); only more SEEDS would.
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

from lategame.eval.arena import build_player, evaluate_built, policy_agent
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


def verdict_of(diff: float, significant: bool) -> str:
    """Three-way verdict. Build 23: the two-way version could not say REGRESSION.

    Through Build 23 this read ``"WIN" if significant and diff > 0 else "NULL"``, which stamped
    Build 23's CI-disjoint p = 0.0003 REVERSAL as ``NULL`` -- indistinguishable from Build 22's
    p = 0.67 nothing. A significant negative effect is a finding, not an absence of one, and on the
    strength axis it has so far been the LARGER of the two the project has measured.
    """
    if not significant:
        return "NULL"
    return "WIN" if diff > 0 else "REGRESSION"


def seed_level_stats(
    base: list[dict[str, Any]], cand: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Paired-by-seed view of a contrast -- the SECOND inference level, REPORTED not adjudicated.

    The pooled z-test above treats all 900+ battles as iid. The SEEDS ARE NOT. Build 23 measured
    per-seed rates of 0.597/0.470/0.607 (v23b) against a within-seed binomial sd of 0.0287 at
    n=300, so the between-seed sd is ~0.0706 -- 2.5x the within-seed noise. That makes the pooled
    verdict a claim about THESE CHECKPOINTS, not about the training procedure, and it means raising
    ``--n`` barely moves the procedure-level SE (0.0622 -> 0.0579 going from n=300 to n=5000).
    Both facts are invisible in the pooled number, so they are written into the artifact here.

    ``t`` is reported for CALIBRATION against the pooled ``z``, deliberately WITHOUT a p-value:
    scipy is not in the env, and a second p-value would need its own multiplicity correction and
    would read as a second pre-registered test, which it is not. ``sign_test_p`` IS exact (a
    two-sided binomial at 0.5, stdlib ``math.comb``) and is the honest cheap check on direction.

    Returns ``None`` when the arms cannot be paired -- unequal seed counts, or fewer than 2 seeds.
    Pairing is only meaningful because an arm's checkpoints come from ``best_checkpoints`` in the
    gate JSON's record order, i.e. seed order, and Build 24's arms share seeds AND init.
    """
    if len(base) != len(cand) or len(base) < 2:
        return None
    a_rates = [float(c["rate"]) for c in base]
    b_rates = [float(c["rate"]) for c in cand]
    diffs = [b - a for a, b in zip(a_rates, b_rates, strict=True)]
    k = len(diffs)
    mean = sum(diffs) / k
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (k - 1))
    se = sd / math.sqrt(k)
    n_pos = sum(1 for d in diffs if d > 0)
    # Two-sided exact sign test at p=0.5, ignoring exact ties (none occur at these sample sizes).
    tail = sum(math.comb(k, i) for i in range(min(n_pos, k - n_pos) + 1))
    return {
        "baseline_rates": [round(r, 4) for r in a_rates],
        "candidate_rates": [round(r, 4) for r in b_rates],
        "paired_diffs": [round(d, 4) for d in diffs],
        "mean_paired_diff": round(mean, 4),
        "sd_paired_diff": round(sd, 4),
        "t": round(mean / se, 3) if se > 0 else 0.0,
        "seeds": k,
        "n_positive": n_pos,
        "sign_test_p": round(min(1.0, 2.0 * tail / 2**k), 4),
    }


def pairwise_comparisons(
    builds: dict[str, Any], names: list[str], alpha: float = 0.05
) -> list[dict[str, Any]]:
    """Every pair, earlier arm as baseline, so a k-arm factorial reads as k(k-1)/2 tests.

    Kept pure and separate from ``main`` because this is the arithmetic a build's verdict rests
    on, and it must be testable without standing up a Showdown server.
    """
    out: list[dict[str, Any]] = []
    for i, base in enumerate(names):
        for cand in names[i + 1 :]:
            a, b = builds[base], builds[cand]
            test = two_proportion_z(a["wins"], a["n"], b["wins"], b["n"])
            significant = test["p_value"] < alpha
            out.append(
                {
                    "baseline": base,
                    "candidate": cand,
                    "disjoint_ci": b["ci95"][0] > a["ci95"][1] or a["ci95"][0] > b["ci95"][1],
                    "alpha": alpha,
                    "significant": significant,
                    **test,
                    "verdict": verdict_of(test["diff"], significant),
                    # Never overrides the verdict -- see seed_level_stats' docstring.
                    "seed_level": seed_level_stats(
                        a.get("per_checkpoint", []), b.get("per_checkpoint", [])
                    ),
                }
            )
    return out


async def score_build(
    checkpoints: list[str],
    opponent: str,
    battle_format: str,
    team: TeamPool | None,
    n: int,
    loop_penalty: float,
    concurrency: int,
    opponent_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Win-rate of every seed's best checkpoint vs ``opponent``, pooled over seeds.

    ``opponent_checkpoint`` makes the opponent a LEARNED policy rather than a fixed baseline --
    the head-to-head against the arm's own warm start. That contrast is ceiling-INDEPENDENT (both
    sides are learned policies on one format), which is the only readable one when a format's
    ceiling is itself in doubt: Build 27 measured search only against the fixed baseline, reported
    "no effect", and missed that the effect was negative.

    The learner is built with the agent the FORMAT wants (``arena.policy_agent``); ``offrl`` is
    singles-only and is refused outright on doubles.
    """
    learner_name = policy_agent(battle_format)
    per_ckpt: list[dict[str, Any]] = []
    for path in checkpoints:
        learner = build_player(
            learner_name,
            battle_format,
            checkpoint_path=path,
            sample=False,  # eval is greedy, matching the deployed policy
            max_concurrent_battles=concurrency,
            team=team,
            loop_penalty=loop_penalty,
        )
        opp = build_player(
            opponent,
            battle_format,
            checkpoint_path=opponent_checkpoint,
            sample=False,
            max_concurrent_battles=concurrency,
            team=team,
        )
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
        help="build label + its ppo_continue_gate JSON; pass 2+ times -- EVERY PAIR is compared",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="significance level. With k>2 arms the pairwise tests are MULTIPLE COMPARISONS: pass "
        "the pre-registered Bonferroni level (e.g. 0.0125 for 4 planned contrasts), do not read "
        "the default 0.05 off a factorial run",
    )
    ap.add_argument("--opponent", default="heuristic")
    # B6f: score against a learned CHECKPOINT rather than a fixed baseline -- the arm against its
    # own warm start. Omitting it reproduces every prior invocation exactly.
    ap.add_argument(
        "--opponent-checkpoint",
        dest="opponent_checkpoint",
        default=None,
        help="checkpoint for --opponent, making it a learned policy (e.g. the warm start)",
    )
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
                ckpts, args.opponent, args.format, team, args.n, args.loop_penalty,
                args.concurrency, args.opponent_checkpoint,
            )
        )

    result: dict[str, Any] = {
        "gate": "seed_strength",
        "opponent": args.opponent,
        "opponent_checkpoint": args.opponent_checkpoint,
        "n_per_checkpoint": args.n,
        "format": args.format,
        "loop_penalty": args.loop_penalty,
        "builds": builds,
        "note": (
            "Every seed's best checkpoint is scored and POOLED -- the across-seed argmax of the "
            "old protocol is dropped, and the sample triples. Per-seed 'best iter' is still a max "
            "over 50 noisy curve evals, so absolute rates stay optimistic; the protocol is "
            "identical across builds, so that bias cancels in the DIFFERENCE, which is what is "
            "under test. CI overlap is a CONSERVATIVE test -- read the z-test too. "
            "TWO INFERENCE LEVELS, and they do not agree. The pooled z-test ('diff'/'z'/'p_value') "
            "is the pre-registered verdict and is a claim about THESE CHECKPOINTS. 'seed_level' "
            "is the same contrast paired by seed, and is the claim about the TRAINING "
            "PROCEDURE -- the between-seed sd (~0.0706 in Build 23) is 2.5x the within-seed "
            "binomial noise, so procedure-level evidence is far weaker than the pooled p reads, "
            "and raising --n does not fix it (only more SEEDS would). Report both; the verdict "
            "field reflects the pooled test alone."
        ),
    }

    names = [name for name, _ in args.build]
    comparisons = pairwise_comparisons(builds, names, args.alpha)

    if comparisons:
        result["alpha"] = args.alpha
        result["comparisons"] = comparisons
        # Back-compat: through Build 23 a two-arm run wrote a single ``comparison`` object, and the
        # four historical results/seed_strength_gate_v2*.json are read in that shape.
        if len(comparisons) == 1:
            result["comparison"] = comparisons[0]

    for c in comparisons:
        a, b = builds[c["baseline"]], builds[c["candidate"]]
        print(f"\n=== {c['baseline']} -> {c['candidate']} ===")
        print(
            f"  {a['rate']:.3f} [{a['ci95'][0]:.3f},{a['ci95'][1]:.3f}]  ->  "
            f"{b['rate']:.3f} [{b['ci95'][0]:.3f},{b['ci95'][1]:.3f}]"
        )
        print(
            f"  diff {c['diff']:+.3f}   z {c['z']:+.2f}   p {c['p_value']:.4f}   "
            f"disjoint_ci {c['disjoint_ci']}   alpha {c['alpha']}"
        )
        s = c.get("seed_level")
        if s:
            print(
                f"  seed-level  diffs [{','.join(f'{d:+.3f}' for d in s['paired_diffs'])}]"
                f"  mean {s['mean_paired_diff']:+.3f}  t {s['t']:+.2f}"
                f"  {s['n_positive']}/{s['seeds']} positive  sign p {s['sign_test_p']:.3f}"
            )
        print(f"  VERDICT {c['verdict']}")

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""B6f Stage A gate: does the doubles loop close, and what does VGC look like once it has?

THE FINDING THIS EXISTS TO CLOSE. `data/vgc_rl.npz` -- the shard B6e's AWR was trained on -- is
140,848 turns over 899 episodes, and it is not what it looks like:

    turns/episode, median            7        (OU shard:  19)
    turns/episode, max          12,795        (OU shard: 205)
    top 10 episodes           33.7% of shard  (OU shard: 1.4%)
    top 100 episodes          94.2% of shard  (OU shard: 7.7%)

The single longest episode carries 12,795 recorded turns over SEVEN unique observation vectors:
the same states re-requested thousands of times. 51.9% of all turns are one signature -- slot 0's
legal set is {pass}, slot 1's is two switches -- which is an absorbing forced-replacement cycle.
96.6% of all rewards are exactly zero, so the value support `_support_from_returns` fitted
(v_min/v_max = -7.175/7.256) was fitted against a near-constant target.

Split by episode length the picture inverts, and this is the part that matters for the campaign:

    non-loop episodes (<=40 turns):  5,553 turns, decision density 0.906, 14.71 legal/slot
    whole shard:                   140,848 turns, decision density 0.571,  2.62 legal/slot

So VGC doubles is a NORMAL, decision-dense format. The "98.4% of recorded turns had exactly one
legal action per slot" note in `data.collect._is_learnable` is an artifact of the loop, not a
property of doubles -- and the B6d/B6e numbers rest on the contaminated distribution.

Root cause: Build 14's `LoopGuard` was never ported. `eval.arena._LOOP_GUARD_AGENTS` excludes
`doubles`, and `LoopGuard.penalty_vector` is a 26-wide vector penalizing indices 0-5 while the
doubles layout is per slot with 0 = pass and 1-6 = switches. The fix is two-part, and the second
part is not optional: the guard is a SOFT penalty, and in the dominant stuck state every legal
option is a switch, so there is nothing to push the argmax toward. A hard per-battle decision
ceiling (`agents.turn_cap`) is what terminates it.

    bash scripts/run_server.sh                       # local Showdown on :8000
    python scripts/vgc_loop_probe.py \
        --format gen9vgc2025regi \
        --team-pool lategame/teambuilding/data/teams_gen9vgc.packed \
        --n 12 --out results/vgc_loop_probe.json

PRE-REGISTERED GATE. Both clauses, and both are reported whether or not they bind:

  1. CONCENTRATION -- the capped arm's top-DECILE turn share must be <= 0.45. Scale-free
     (uniform = 0.10), unlike "the top 10 episodes", which with E episodes cannot fall below
     10/E and so measures the sample size. Measured: OU 0.239, uncapped VGC 0.922.
  2. REPETITION -- unique observations per episode must average >= 0.90 of its turns. No length
     statistic can see this: a 39-turn cycle under a 40-turn ceiling clears every concentration
     bar and is still a loop.

Anything else is booked as PARTIAL rather than rounded up to whichever clause passed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np

from lategame.config import VGC_FORMAT

_DEFAULT_POOL = "random,maxbasepower,simpleheuristics,heuristic"


def shard_stats(
    obs: np.ndarray, mask: np.ndarray, reward: np.ndarray, done: np.ndarray
) -> dict[str, Any]:
    """Loop + composition diagnostics for one collected shard.

    Pure (no server, no poke-env) because this is the arithmetic the Stage A verdict rests on,
    and it must be testable without standing up Showdown -- the same rule
    ``scripts/seed_strength_gate.pairwise_comparisons`` follows.
    """
    n = int(len(done))
    if n == 0:
        raise ValueError("empty shard: nothing to measure")
    episode = np.cumsum(np.concatenate([[0], done[:-1]])).astype(np.int64)
    lengths = np.bincount(episode)
    order = np.argsort(-lengths)
    cumulative = np.cumsum(lengths[order])

    # UNIQUE OBSERVATIONS PER EPISODE is the load-bearing statistic, not episode length. A loop is
    # states REPEATING; an episode can be long and healthy, and a short cycle under the ceiling
    # would pass a length test while still contributing nothing but duplicated frames.
    uniq_ratio = []
    for e in order[: min(len(lengths), 25)]:
        rows = obs[episode == e]
        uniq_ratio.append(len(np.unique(rows, axis=0)) / max(1, len(rows)))

    # TOP-DECILE share, not top-TEN: "the 10 longest episodes" is not comparable across shards,
    # because with E episodes it cannot fall below 10/E. The OU shard has 5,503 episodes and the
    # VGC one 899, so a fixed top-10 bar would have measured the sample size. The decile share is
    # scale-free (uniform = 0.10) and separates the two cleanly -- see the bars below.
    decile = max(1, len(lengths) // 10)
    sorted_lengths = np.sort(lengths).astype(float)
    m = len(sorted_lengths)
    gini = float(
        (2 * np.arange(1, m + 1) - m - 1).dot(sorted_lengths) / (m * sorted_lengths.sum())
    )

    legal = mask.sum(axis=-1)
    has_choice = (legal.max(axis=1) > 1) if mask.ndim == 3 else (legal > 1)
    n_choice = int(has_choice.sum())
    median = float(np.median(lengths))
    return {
        "n_turns": n,
        "n_episodes": int(len(lengths)),
        "turns_per_episode_mean": round(float(lengths.mean()), 2),
        "turns_per_episode_median": median,
        "turns_per_episode_max": int(lengths.max()),
        "max_over_median_length": round(float(lengths.max() / max(1.0, median)), 1),
        "top_decile_share": round(float(cumulative[decile - 1] / n), 4),
        "length_gini": round(gini, 4),
        "top1_episode_share": round(float(cumulative[0] / n), 4),
        "top10_episode_share": round(float(cumulative[min(9, len(cumulative) - 1)] / n), 4),
        "top100_episode_share": round(float(cumulative[min(99, len(cumulative) - 1)] / n), 4),
        "unique_obs_ratio_min": round(float(min(uniq_ratio)), 4),
        "unique_obs_ratio_mean": round(float(np.mean(uniq_ratio)), 4),
        "decision_frac": round(float(has_choice.mean()), 4),
        "n_decision": n_choice,
        "mean_legal_per_slot_on_decisions": (
            round(float(legal[has_choice].mean()), 3) if n_choice else 0.0
        ),
        "zero_reward_frac": round(float((reward == 0).mean()), 4),
    }


# Stage A's pre-registered thresholds. Both are SET FROM THE TWO REAL SHARDS rather than chosen,
# measured 2026-08-15:
#
#                        OU (healthy)   VGC (loop)   uniform
#     top-decile share       0.239        0.922        0.10
#     length Gini            0.304        0.901        0.0
#     max/median length       10.8       1827.9        1.0
#
#: Midway between the two on the concentration axis, and comfortably clear of OU's value so a
#: healthy shard with more length variance than OU's still passes.
TOP_DECILE_BAR = 0.45
#: States REPEATING is what no length statistic can see: a 39-turn cycle under a 40-turn ceiling
#: would clear every concentration bar above and still be a loop.
UNIQUE_OBS_BAR = 0.90


def verdict(arms: dict[str, dict[str, Any]], baseline: str, fixed: str) -> dict[str, Any]:
    """Adjudicate Stage A against the thresholds above. Reports BOTH clauses, always."""
    b, f = arms[baseline], arms[fixed]
    share_ok = f["top_decile_share"] <= TOP_DECILE_BAR
    uniq_ok = f["unique_obs_ratio_mean"] >= UNIQUE_OBS_BAR
    return {
        "baseline_arm": baseline,
        "fixed_arm": fixed,
        "top_decile_share_before": b["top_decile_share"],
        "top_decile_share_after": f["top_decile_share"],
        "top_decile_share_bar": TOP_DECILE_BAR,
        "top_decile_share_ok": bool(share_ok),
        "length_gini_before": b["length_gini"],
        "length_gini_after": f["length_gini"],
        "unique_obs_ratio_before": b["unique_obs_ratio_mean"],
        "unique_obs_ratio_after": f["unique_obs_ratio_mean"],
        "unique_obs_bar": UNIQUE_OBS_BAR,
        "unique_obs_ok": bool(uniq_ok),
        "decision_frac_before": b["decision_frac"],
        "decision_frac_after": f["decision_frac"],
        # Both clauses are reported whether or not they bind, so a partial fix is booked as one
        # rather than rounded up to the clause that happened to pass.
        "verdict": "LOOP_CLOSED" if (share_ok and uniq_ok) else "PARTIAL",
    }


async def run_probe(
    pool: list[str],
    n_per_pair: int,
    fmt: str,
    team_pool: str | None,
    arms: list[dict[str, Any]],
    out: str,
) -> dict[str, Any]:
    from lategame.data.collect import collect_trajectories

    measured: dict[str, dict[str, Any]] = {}
    for arm in arms:
        label = str(arm["label"])
        print(f"\n=== arm {label}: {arm} ===")
        ds = await collect_trajectories(
            pool,
            n_per_pair,
            fmt,
            None,
            0.99,
            team_pool=team_pool,
            max_battle_turns=arm["max_battle_turns"],
            loop_penalty=float(arm["loop_penalty"]),
        )
        measured[label] = {
            **{k: arm[k] for k in ("max_battle_turns", "loop_penalty")},
            **shard_stats(ds.obs, ds.mask, ds.reward, ds.done),
        }
        s = measured[label]
        print(
            f"  turns {s['n_turns']} / {s['n_episodes']} eps | max {s['turns_per_episode_max']} | "
            f"top_decile {s['top_decile_share']:.3f} gini {s['length_gini']:.3f} | "
            f"uniq_obs {s['unique_obs_ratio_mean']:.3f} | "
            f"decision_frac {s['decision_frac']:.3f} | zero_reward {s['zero_reward_frac']:.3f}"
        )

    labels = list(measured)
    result: dict[str, Any] = {
        "gate": "vgc_loop_probe",
        "format": fmt,
        "pool": pool,
        "n_per_pair": n_per_pair,
        "team_pool": team_pool,
        "arms": measured,
        "prior_shard": {
            "path": "data/vgc_rl.npz",
            "note": (
                "measured 2026-08-15 on the shard B6e trained from: 140,848 turns / 899 episodes, "
                "top10 share 0.337, top100 share 0.942, max 12,795 turns over 7 unique obs, "
                "top_decile share 0.922 / Gini 0.901 against the OU shard's 0.239 / 0.304, "
                "decision_frac 0.571, zero_reward_frac 0.966"
            ),
            "top_decile_share": 0.9219,
            "length_gini": 0.9014,
            "top10_episode_share": 0.337,
            "top100_episode_share": 0.942,
            "turns_per_episode_max": 12795,
            "decision_frac": 0.5707,
            "zero_reward_frac": 0.9659,
        },
    }
    if len(labels) >= 2:
        result["stage_a"] = verdict(measured, labels[0], labels[-1])
        print(f"\nSTAGE A: {result['stage_a']['verdict']}")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default=_DEFAULT_POOL, help="Comma-separated demonstrators")
    parser.add_argument("--n", type=int, default=12, help="Battles per distinct pair, per arm")
    parser.add_argument("--format", dest="battle_format", default=VGC_FORMAT)
    parser.add_argument(
        "--team-pool",
        dest="team_pool",
        default="lategame/teambuilding/data/teams_gen9vgc.packed",
        help="Packed-team pool; required for a teambuilt format",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=1200,
        help="Decision ceiling for the FIXED arm (agents.turn_cap.DEFAULT_MAX_BATTLE_TURNS)",
    )
    # The guard is a no-op for the poke-env baselines (they carry none), so it only moves the
    # measurement when the pool includes `doubles`. Reported either way rather than assumed.
    parser.add_argument(
        "--loop-penalty",
        dest="loop_penalty",
        type=float,
        default=0.0,
        help="DoublesLoopGuard penalty for the fixed arm; only affects learned doubles agents",
    )
    parser.add_argument("--out", default="results/vgc_loop_probe.json")
    args = parser.parse_args(argv)

    pool = [name.strip() for name in args.pool.split(",") if name.strip()]
    arms = [
        # The baseline reproduces the shard above: no ceiling, no guard.
        {"label": "uncapped", "max_battle_turns": None, "loop_penalty": 0.0},
        {"label": "capped", "max_battle_turns": args.cap, "loop_penalty": args.loop_penalty},
    ]
    asyncio.run(
        run_probe(pool, args.n, args.battle_format, args.team_pool, arms, args.out)
    )


if __name__ == "__main__":
    main()

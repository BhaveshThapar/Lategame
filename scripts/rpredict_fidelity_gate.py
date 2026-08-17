"""Lever 11 gate A: forward-model fidelity -- the cheap KILL gate for R-PREDICT.

Lever 10 (PPO continuation) was AMBER: on-policy gradient is near a local ceiling for
gen9-RB vs the heuristic. Its verdict named the successor -- R-PREDICT: depth-limited
search / opponent-modeling on the strong base policy + working value head. Search needs a
*forward model* (step a hypothetical (my-move, opp-move) -> next state), which this repo
does not have: the net gives V(s) and pi(a|s) but not Q(s,a), and the vendored simulator
is wired only for replay re-simulation. We build the forward model on the simulator's
``State.serializeBattle``/``deserializeBattle`` (fork) + ``battle.choose`` (step).

Before building any search on top of that primitive, prove it is FAITHFUL. This gate
replays real cached replays turn-by-turn and, at each decision round, forks the battle,
steps the fork with the same choices, and checks the result matches stepping the battle
directly. The serialized state carries the PRNG seed, so a faithful fork is an *exact*
match (same damage rolls / crits / accuracy) -- a high mismatch rate is real infidelity.

    python scripts/rpredict_fidelity_gate.py --limit 300 --out results/rpredict_fidelity.json

No local server and no training -- it only forks the vendored simulator over replays.
Gate: PASS (build search) if the core match rate (hp/status/faint/field/hazards/active)
>= --threshold (default 0.99); KILL (pivot to the curriculum fallback) otherwise.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rotomai.data.replays import cached_replay_paths
from rotomai.search.fidelity import run_fidelity_files

_PASS_THRESHOLD = 0.99


def run_gate(
    cache_dir: str,
    limit: int,
    seed: int,
    threshold: float,
    out: str,
    showdown_dir: str,
    node: str,
) -> dict:
    paths = cached_replay_paths(cache_dir)
    if not paths:
        raise SystemExit(f"no cached replays under '{cache_dir}' -- run the scraper first.")
    if 0 < limit < len(paths):
        rng = random.Random(seed)
        paths = rng.sample(paths, limit)
    print(f"fidelity gate: {len(paths)} replays (seed={seed}), threshold={threshold:.3f}")

    stats = run_fidelity_files(paths, showdown_dir=showdown_dir, node=node)
    verdict = "PASS" if stats.core_match_rate >= threshold else "KILL"

    result = {
        "gate": "rpredict_fidelity",
        "cache_dir": cache_dir,
        "limit": limit,
        "seed": seed,
        "threshold": threshold,
        "verdict": verdict,
        **stats.to_dict(),
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"\nreplays={stats.replays} (errored={stats.errored})  transitions={stats.transitions}\n"
        f"core  match rate = {stats.core_match_rate:.4f}  (mismatch={stats.core_mismatch})\n"
        f"full  match rate = {stats.full_match_rate:.4f}  (mismatch={stats.full_mismatch})\n"
        f"drive_errors     = {stats.drive_errors}\n"
        f"VERDICT: {verdict}  (>= {threshold:.3f} core => build search; else KILL)"
    )
    if stats.samples:
        print(f"\nfirst {len(stats.samples)} mismatch sample(s):")
        for s in stats.samples:
            print("  " + json.dumps(s)[:400])
    print(f"\nwrote {out_path}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rpredict_fidelity_gate")
    parser.add_argument("--cache-dir", default="replays/gen9randombattle")
    parser.add_argument("--limit", type=int, default=300, help="0 = all cached replays")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed when --limit < total")
    parser.add_argument("--threshold", type=float, default=_PASS_THRESHOLD)
    parser.add_argument("--out", default="results/rpredict_fidelity.json")
    parser.add_argument("--showdown-dir", default="third_party/pokemon-showdown")
    parser.add_argument("--node", default="node")
    args = parser.parse_args(argv)

    result = run_gate(
        args.cache_dir,
        args.limit,
        args.seed,
        args.threshold,
        args.out,
        args.showdown_dir,
        args.node,
    )
    raise SystemExit(0 if result["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()

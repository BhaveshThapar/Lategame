"""Lever 11 mini-gate: state-reconstruction fidelity for R-PREDICT.

Gate A proved the forward *step* is exact. This is the other half: does determinizing a live
POV into a full Showdown battle preserve the *observable* state? We re-simulate replays, snapshot
each decision turn, determinize it, and compare the reconstruction's observable digest to
poke-env's (our full team + revealed opponent mons + field + hazards; hidden filled slots are not
checked). Cheap -- no live server, no training.

    python scripts/rpredict_recon_gate.py --limit 40 --out results/rpredict_recon.json

Gate (informational, not a hard kill): a high match rate means search will fork from a faithful
root; low fields point at determinization bugs to fix before spending live-eval compute.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rotomai.data.replays import iter_cached_replays
from rotomai.search.recon_check import run_recon_check


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rpredict_recon_gate")
    parser.add_argument("--cache-dir", default="replays/gen9randombattle")
    parser.add_argument("--limit", type=int, default=40, help="replays to sample (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=0, help="cap snapshots per side (0=all)")
    parser.add_argument("--out", default="results/rpredict_recon.json")
    parser.add_argument("--showdown-dir", default="third_party/pokemon-showdown")
    parser.add_argument("--node", default="node")
    args = parser.parse_args(argv)

    replays = list(iter_cached_replays(args.cache_dir))
    if not replays:
        raise SystemExit(f"no cached replays under '{args.cache_dir}'.")
    if 0 < args.limit < len(replays):
        replays = random.Random(args.seed).sample(replays, args.limit)
    print(f"reconstruction mini-gate: {len(replays)} replays (seed={args.seed})")

    stats = run_recon_check(replays, args.showdown_dir, args.node, args.max_turns)
    result = {"gate": "rpredict_recon", "limit": args.limit, "seed": args.seed, **stats.to_dict()}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"\nsnapshots={stats.snapshots} (errored={stats.errored})  checks={stats.checks}\n"
        f"overall observable match rate = {stats.match_rate:.4f}  (mismatch={stats.mismatches})"
    )
    print("\nper-field:")
    for k, v in result["by_field"].items():
        print(f"  {k:10s} rate={v['rate']:.4f}  ({v['mismatch']}/{v['total']} mismatched)")
    if stats.samples:
        print("\nsample mismatches:")
        for s in stats.samples:
            print("  " + json.dumps(s)[:300])
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

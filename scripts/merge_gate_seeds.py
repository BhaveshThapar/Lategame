"""Merge per-seed ppo_continue_gate JSONs into the multi-seed gate file the verdict reads.

Seeds are run ONE AT A TIME (a wide net needs ~900 MB of checkpoints and the box has ~20 GB of
headroom), so each writes its own `..._s{N}.json` holding a single record. But
`seed_strength_gate.py` -- the authoritative WIN/NULL comparison -- reads
`gate["records"][*]["best_checkpoint"]` and scores EVERY seed's best checkpoint, pooling them to
900 battles/arm. Hand it a file with one record and it silently pools 300 instead: the z-test
still runs, still prints a verdict, and has quietly lost the power the protocol was designed for.

v19 and v20 were assembled by hand. This encodes the rule instead, because the failure mode above
is invisible in the output.

The merge is mechanical except for two fields that are inputs, not derivations:

    --ladder-source   the per-seed `--ladder-n 20` ladders are throwaways (n=20 resolves nothing);
                      the merged file points at the authoritative n=300 ladder instead.
    --note            the build note: what the lever was, and the trust-region certificate that
                      says whether it landed cleanly (see scripts/ppo_telemetry.py).

Verified by round-tripping v20: rebuilding results/ppo_ou_gate_v20.json from its three per-seed
files reproduces the committed file byte for byte (tests/test_merge_gate_seeds.py).

    python scripts/merge_gate_seeds.py \
      --seed-json results/ppo_ou_gate_v21_s0.json \
      --seed-json results/ppo_ou_gate_v21_s1.json \
      --seed-json results/ppo_ou_gate_v21_s2.json \
      --ladder-source "authoritative n=300 in results/seed_strength_gate_v21.json" \
      --note "Build 21: ..." \
      --out results/ppo_ou_gate_v21.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

# Fields the seeds must agree on: they define the ARM, so a mismatch means the per-seed files came
# from different experiments and pooling them would compare a build against itself. `seed`-varying
# and outcome fields are excluded on purpose.
ARM_FIELDS = (
    "init",
    "arm",
    "format",
    "team_pool",
    "loop_penalty",
    "ckpt_prefix",
    "ent_coef",
    "ent_coef_final",
    "lr",
    "lr_final",
    "iters",
    "games_per_opp",
    "eval_n",
    "ladder_n",
    "opponent",
)


def merge(seed_gates: list[dict[str, Any]], ladder_source: str, note: str) -> dict[str, Any]:
    """Merged gate file, keeping the per-seed key order (v19/v20 shape) with ``_note`` appended."""
    if not seed_gates:
        raise ValueError("no per-seed gate files given")

    first = seed_gates[0]
    for other in seed_gates[1:]:
        mismatched = {
            f: (first.get(f), other.get(f)) for f in ARM_FIELDS if first.get(f) != other.get(f)
        }
        if mismatched:
            raise ValueError(
                "per-seed files disagree on the arm -- refusing to pool different experiments: "
                f"{mismatched}"
            )

    records = [r for gate in seed_gates for r in gate["records"]]
    seeds = [r["seed"] for r in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seed in the inputs: {seeds}")

    bests = [r["best_vs_heuristic"] for r in records]

    # Start from seed 0's scaffold so key ORDER matches v19/v20; assigning an existing key keeps
    # its position, so only `_note` lands at the end.
    best = max(records, key=lambda r: r["best_vs_heuristic"])
    merged: dict[str, Any] = dict(first)
    merged["seeds"] = seeds
    merged["best_checkpoint"] = best["best_checkpoint"]
    merged["confirmatory_ladder"] = {"_source": ladder_source}
    merged["records"] = records
    merged["summary"] = {
        "best_vs_heuristic_mean": round(statistics.fmean(bests), 4),
        "best_vs_heuristic_std": round(statistics.pstdev(bests), 4),
        "best_vs_heuristic_per_seed": bests,
        "best_iters": [r["best_iter"] for r in records],
        "final_vs_iter0": [r["final_vs_iter0"] for r in records],
    }
    merged["_note"] = note
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seed-json",
        action="append",
        required=True,
        metavar="PATH",
        help="a per-seed ppo_continue_gate JSON; pass once per seed, in seed order",
    )
    ap.add_argument(
        "--ladder-source", required=True, help="pointer to the authoritative n=300 ladder"
    )
    ap.add_argument("--note", required=True, help="build note, incl. the trust-region certificate")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gates = [json.loads(Path(p).read_text()) for p in args.seed_json]
    merged = merge(gates, args.ladder_source, args.note)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    s = merged["summary"]
    print(f"merged {len(merged['records'])} seeds {merged['seeds']} -> {out}")
    for r in merged["records"]:
        print(
            f"  seed {r['seed']}  best iter {r['best_iter']:>2}  "
            f"{r['best_vs_heuristic']:.3f}  {r['best_checkpoint']}"
        )
    print(f"  mean {s['best_vs_heuristic_mean']:.4f} +/- {s['best_vs_heuristic_std']:.4f}")
    print(
        "  best_checkpoint (across-seed argmax, NOT what seed_strength_gate scores): "
        f"{merged['best_checkpoint']}"
    )


if __name__ == "__main__":
    main()

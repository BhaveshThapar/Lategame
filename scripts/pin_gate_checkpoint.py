"""Build 25: rewrite a merged gate file so every seed is scored at a PINNED iteration.

`seed_strength_gate.py` scores `records[*]["best_checkpoint"]` -- each seed's `argmax` over its
noisy `eval_n`=100 curve. Through Build 24 that selection step was harmless in the DIFFERENCE,
because it was common to both arms: Build 24 simulated its 50-vs-80 comparison at **+0.0014** and
cleared it.

**It stops cancelling when the arms differ in LENGTH.** A 160-iter arm draws twice as many samples
at that `argmax` as an 80-iter arm, so it lands on a truly-better checkpoint more often even if the
two policies are identical in expectation. Simulated for Build 25 (true late-window p ~ N(mu,
sigma_b), observed ~ Bin(100,p)/100, 200k trials, sigma_b = 0.0428 estimated from the Build 23/24
curves by subtracting the binomial component from the observed late-window variance), the
differential is **+0.0045 (80->120) / +0.0028 (120->160) / +0.0073 (80->160)** -- roughly 30% of
the smallest dose Build 25 is powered to call real.

So Build 25 reads its contrasts twice. This script produces the second read: pin every seed to the
run's TERMINAL iteration, where there is no selection at all, and hand the result to the
UNMODIFIED authoritative gate. The pinned read is descriptive, not a second family of tests -- but
a SIGN DISAGREEMENT between the two reads is itself the finding, and is to be booked as one.

The output is deliberately self-describing (`_pinned_iter`, and a rewritten `_note`): a file whose
`best_checkpoint` is not a best would otherwise be indistinguishable from a seed-best gate, which
is exactly the class of invisible failure `merge_gate_seeds.py` exists to prevent.

    python scripts/pin_gate_checkpoint.py \
      --gate results/ppo_ou_gate_v25b.json \
      --out results/ppo_ou_gate_v25b_terminal.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def checkpoint_for(out_dir: str, init: str, iteration: int) -> str:
    """Mirror of `ppo_continue_gate._best_checkpoint` -- iteration 0 is the warm start itself.

    Not imported from there: that module pulls in torch through `rotomai.train.ppo`, and this is
    a pure JSON rewrite. `pin` cross-checks the format against the record it is rewriting, so a
    drift in the naming convention fails loudly instead of producing a path to nothing.
    """
    if iteration == 0:
        return init
    return str(Path(out_dir) / f"iter_{iteration:02d}.pt")


def curve_point(record: dict[str, Any], iteration: int) -> dict[str, Any]:
    for point in record["curve"]:
        if point["iter"] == iteration:
            return point
    raise ValueError(
        f"seed {record['seed']} has no curve point at iteration {iteration} "
        f"(curve runs {record['curve'][0]['iter']}..{record['curve'][-1]['iter']})"
    )


def pin(gate: dict[str, Any], iteration: int | None = None) -> dict[str, Any]:
    """Gate file with every seed's scored checkpoint pinned to ``iteration`` (default: terminal).

    Raises when the iteration is missing from any seed's curve -- a pin past the end of a run
    would otherwise name a checkpoint that was never written, and the strength gate's preflight
    would report it as the gitignored-checkpoints problem, which it is not.
    """
    iteration = gate["iters"] if iteration is None else iteration
    init = gate["init"]

    records = []
    for record in gate["records"]:
        # Drift guard: rebuild the path this record ALREADY carries. If the convention moved, the
        # pinned path would be wrong in exactly the same way, silently.
        rebuilt = checkpoint_for(record["out_dir"], init, record["best_iter"])
        if rebuilt != record["best_checkpoint"]:
            raise ValueError(
                "checkpoint naming has drifted from ppo_continue_gate._best_checkpoint: "
                f"rebuilt {rebuilt!r} for seed {record['seed']}'s best_iter "
                f"{record['best_iter']}, but the record says {record['best_checkpoint']!r}"
            )

        point = curve_point(record, iteration)
        pinned = dict(record)
        pinned["best_iter"] = iteration
        pinned["best_vs_heuristic"] = point["vs_heuristic"]
        pinned["best_checkpoint"] = checkpoint_for(record["out_dir"], init, iteration)
        pinned["delta_vs_start"] = round(point["vs_heuristic"] - record["start_vs_heuristic"], 4)
        records.append(pinned)

    rates = [r["best_vs_heuristic"] for r in records]
    out: dict[str, Any] = dict(gate)
    out["records"] = records
    out["best_checkpoint"] = max(records, key=lambda r: r["best_vs_heuristic"])["best_checkpoint"]
    out["summary"] = {
        "best_vs_heuristic_mean": round(statistics.fmean(rates), 4),
        "best_vs_heuristic_std": round(statistics.pstdev(rates), 4),
        "best_vs_heuristic_per_seed": rates,
        "best_iters": [r["best_iter"] for r in records],
        "final_vs_iter0": [r["final_vs_iter0"] for r in records],
    }
    out["_pinned_iter"] = iteration
    out["_note"] = (
        f"PINNED to iteration {iteration} by scripts/pin_gate_checkpoint.py -- "
        "'best_checkpoint' is NOT a best here, it is a fixed iteration, so this file carries NO "
        "argmax-selection bias and must not be compared against a seed-best gate file. "
        f"Original note: {gate.get('_note', '')}"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", required=True, help="a MERGED multi-seed ppo_continue_gate JSON")
    ap.add_argument(
        "--iter",
        type=int,
        default=None,
        dest="iteration",
        help="iteration to pin (default: the run's terminal iteration, i.e. the gate's `iters`)",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gate = json.loads(Path(args.gate).read_text())
    pinned = pin(gate, args.iteration)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pinned, indent=2), encoding="utf-8")

    s = pinned["summary"]
    print(f"pinned {len(pinned['records'])} seeds to iteration {pinned['_pinned_iter']} -> {out}")
    for r in pinned["records"]:
        print(f"  seed {r['seed']}  {r['best_vs_heuristic']:.3f}  {r['best_checkpoint']}")
    print(f"  mean {s['best_vs_heuristic_mean']:.4f} +/- {s['best_vs_heuristic_std']:.4f}")


if __name__ == "__main__":
    main()

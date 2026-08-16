"""Identity-prior BC gate: embeddings OFF vs random-init vs dex-prior warm-start.

The data-scaling sweep (docs/RESULTS.md) left identity embeddings *starved but emerging* --
at the exhausted replay ceiling, random-init ON (0.401) trailed OFF (0.418) by ~1.7pp. This
gate tests the cheap remedy: warm-start the species/move embeddings from dex-feature priors
(``features.embed_prior``) so they don't have to learn identity from scratch.

Three arms per rung -- OFF (``id_embed=False``), ON-random (random init), ON-prior (dex
warm-start) -- across seeds, on the same nested subsets as ``embed_scaling_sweep.py``.
Reports mean +/- std val-acc and both gaps (prior - random, prior - off). The decision gate:
ON-prior clears OFF at the ceiling (non-overlapping bands) -> GREEN, greenlight an offrl/PPO
retrain; ON-prior >= random but <= OFF -> AMBER; ON-prior ~= random -> RED (priors dead).

    python scripts/embed_prior_sweep.py \
        --data data/resim_v3_gen9rb_bc.npz \
        --out results/embed_prior_sweep.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from lategame.model.factory import MODEL_ENTITY_TRANSFORMER
from lategame.train.bc import TrainConfig, train_bc

# Focused on the emergence -> ceiling region from the scaling sweep (full grid x 3 arms is
# costly); the ceiling rung is always included via the shard total.
_RUNGS = (8000, 16000, 32000)

# (label, id_embed, id_embed_init) -- the three controlled arms.
_ARMS = (
    ("off", False, "random"),
    ("random", True, "random"),
    ("prior", True, "prior"),
)


def _default_points(total: int) -> list[int]:
    return sorted({n for n in _RUNGS if n < total} | {total})


def run_sweep(
    data: str,
    out: str,
    seeds: list[int],
    points: list[int] | None,
    epochs: int,
    device: str,
    id_embed_dim: int,
) -> dict:
    total = int(np.load(data)["obs"].shape[0])
    points = points or _default_points(total)
    print(f"shard {data}: {total} samples | points {points} | seeds {seeds}")

    tmp = Path(tempfile.mkdtemp(prefix="embed_prior_sweep_"))
    records: list[dict] = []
    for n in points:
        for label, id_embed, init in _ARMS:
            accs: list[float] = []
            for seed in seeds:
                cfg = TrainConfig(
                    epochs=epochs,
                    device=device,
                    model_type=MODEL_ENTITY_TRANSFORMER,
                    id_embed=id_embed,
                    id_embed_dim=id_embed_dim,
                    id_embed_init=init,
                    seed=seed,
                    max_samples=n,
                )
                t0 = time.time()
                metrics = train_bc(data, tmp / "sweep.pt", cfg)
                accs.append(float(metrics["val_acc"]))
                print(
                    f"N={n:>7} arm={label:<6} seed={seed} "
                    f"val_acc={metrics['val_acc']:.4f} ({time.time() - t0:.0f}s)"
                )
            records.append(
                {
                    "n": n,
                    "arm": label,
                    "seeds": seeds,
                    "val_acc": accs,
                    "mean": statistics.mean(accs),
                    "std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                }
            )

    result = {
        "data": data,
        "total_samples": total,
        "points": points,
        "seeds": seeds,
        "epochs": epochs,
        "id_embed_dim": id_embed_dim,
        "arms": [a[0] for a in _ARMS],
        "records": records,
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print_table(records)
    print(f"\nwrote {out_path}")
    return result


def _print_table(records: list[dict]) -> None:
    by_n: dict[int, dict[str, dict]] = {}
    for rec in records:
        by_n.setdefault(rec["n"], {})[rec["arm"]] = rec
    header = (
        f"\n{'N':>8}  {'OFF':>14}  {'RANDOM':>14}  {'PRIOR':>14}  "
        f"{'prior-rand':>11}  {'prior-off':>11}"
    )
    print(header)
    print("-" * len(header))
    for n in sorted(by_n):
        arms = by_n[n]
        off, rand, prior = arms.get("off"), arms.get("random"), arms.get("prior")
        if not (off and rand and prior):
            continue
        print(
            f"{n:>8}  "
            f"{off['mean']:.3f}+-{off['std']:.3f}  "
            f"{rand['mean']:.3f}+-{rand['std']:.3f}  "
            f"{prior['mean']:.3f}+-{prior['std']:.3f}  "
            f"{prior['mean'] - rand['mean']:>+11.3f}  "
            f"{prior['mean'] - off['mean']:>+11.3f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="embed_prior_sweep")
    parser.add_argument("--data", default="data/resim_v3_gen9rb_bc.npz", help="BC shard (.npz)")
    parser.add_argument("--out", default="results/embed_prior_sweep.json")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated model seeds")
    parser.add_argument(
        "--points", default="", help="Comma-separated N rungs (default: 8k/16k/32k + ceiling)"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--id-embed-dim", type=int, default=32)
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    points = [int(p) for p in args.points.split(",") if p.strip()] or None
    run_sweep(args.data, args.out, seeds, points, args.epochs, args.device, args.id_embed_dim)


if __name__ == "__main__":
    main()

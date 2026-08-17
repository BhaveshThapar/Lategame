"""Data-scaling sweep: BC val-acc with identity embeddings ON vs OFF across data sizes.

Trains the EntityTransformer by behaviour cloning on nested subsets of a winners-only
BC shard at several sample counts N, both ``id_embed`` ON and OFF, across seeds. Reports
mean +/- std val-acc per (N, arm) and the OFF-ON gap, to test whether the gap *closes*
as N grows -- the data-starvation hypothesis from docs/RESULTS.md (the R-ENCODE gate showed
ON 26.9% < OFF 36.0% at ~4196 samples; if that was starvation, the gap should shrink with
more data).

    python scripts/embed_scaling_sweep.py \
        --data data/resim_v3_gen9rb_bc.npz \
        --out results/embed_scaling_sweep.json

Writes the full result grid to ``--out`` (JSON) and prints a summary table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from rotomai.model.factory import MODEL_ENTITY_TRANSFORMER
from rotomai.train.bc import TrainConfig, train_bc

# 4196 is the original R-ENCODE gate's sample count -> kept as an anchor for continuity.
_RUNGS = (1000, 2000, 4196, 8000, 16000, 32000, 64000, 128000)


def _default_points(total: int) -> list[int]:
    """Log-spaced rungs below the shard size, always including the full shard."""
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

    tmp = Path(tempfile.mkdtemp(prefix="embed_sweep_"))
    records: list[dict] = []
    for n in points:
        for id_embed in (True, False):
            accs: list[float] = []
            for seed in seeds:
                cfg = TrainConfig(
                    epochs=epochs,
                    device=device,
                    model_type=MODEL_ENTITY_TRANSFORMER,
                    id_embed=id_embed,
                    id_embed_dim=id_embed_dim,
                    seed=seed,
                    max_samples=n,
                )
                t0 = time.time()
                metrics = train_bc(data, tmp / "sweep.pt", cfg)
                accs.append(float(metrics["val_acc"]))
                print(
                    f"N={n:>7} id_embed={int(id_embed)} seed={seed} "
                    f"val_acc={metrics['val_acc']:.4f} ({time.time() - t0:.0f}s)"
                )
            records.append(
                {
                    "n": n,
                    "id_embed": id_embed,
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
        "records": records,
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print_table(records)
    print(f"\nwrote {out_path}")
    return result


def _print_table(records: list[dict]) -> None:
    by_n: dict[int, dict[bool, dict]] = {}
    for rec in records:
        by_n.setdefault(rec["n"], {})[rec["id_embed"]] = rec
    print(f"\n{'N':>8}  {'ON mean+-std':>16}  {'OFF mean+-std':>16}  {'gap(OFF-ON)':>12}")
    print("-" * 60)
    for n in sorted(by_n):
        on, off = by_n[n].get(True), by_n[n].get(False)
        if on is None or off is None:
            continue
        gap = off["mean"] - on["mean"]
        print(
            f"{n:>8}  {on['mean']:.3f}+-{on['std']:.3f}      "
            f"{off['mean']:.3f}+-{off['std']:.3f}      {gap:>+8.3f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="embed_scaling_sweep")
    parser.add_argument("--data", default="data/resim_v3_gen9rb_bc.npz", help="BC shard (.npz)")
    parser.add_argument("--out", default="results/embed_scaling_sweep.json")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated model seeds")
    parser.add_argument(
        "--points", default="", help="Comma-separated N rungs (default: log-spaced from shard)"
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

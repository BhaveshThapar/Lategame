"""Lever 9 gate: AWR offline-RL at data scale, win-rate vs the heuristic.

Retrains the value+AWR offline-RL policy on the full all-turns RL shard (winners AND
losers, with shaped Monte-Carlo returns) and measures win-rate vs the heuristic. This is
the "learning target / data composition" lever from plan.md 13.1: pure winners-only BC
saturates ~0.42 val-acc and never moved win-rate off the ~27-34% plateau; advantage-
weighting can use losers' turns and deviate from imitation, so it is the untested lever
all eight prior levers point at. Every offrl checkpoint on disk predates the 9.65x-scaled
``resim_v3`` shard -- AWR has never seen this much data.

Two arms, each trained per seed then evaluated vs the heuristic at ``--eval-n`` battles:

* ``mlp``      -- flat actor-critic (the exact M3 baseline), warm-started from a fresh
                  MLP BC checkpoint. Isolates data-scale x value-RL with no encoder confound.
* ``et_prior`` -- EntityTransformer + dex-prior identity embeddings (the BC-gate default),
                  trained from scratch (the transformer can't warm-start off the MLP BC).

    bash scripts/run_server.sh                       # local Showdown on :8000
    python scripts/offrl_scale_gate.py \
        --data data/resim_v3_gen9rb_rl.npz \
        --bc-init checkpoints/bc_mlp_resim_v3.pt \
        --out results/offrl_scale_gate.json

Writes the full result grid to ``--out`` (JSON) and prints a summary table. Gate (vs the
27-34% plateau): GREEN if a seed band clears it with non-overlapping bands; AMBER if it
lands in/near the plateau; RED below.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from lategame.config import DEFAULT_FORMAT
from lategame.eval.arena import build_player, evaluate_built
from lategame.model.factory import MODEL_ACTOR_CRITIC, MODEL_ENTITY_TRANSFORMER
from lategame.train.offline_rl import OfflineRLConfig, train_offline_rl

_CKPT_DIR = Path("checkpoints")
_EVAL_CONCURRENCY = 20  # the local server is the bottleneck; keep many battles in flight


def _arm_config(
    arm: str, seed: int, epochs: int, beta: float, device: str, bc_init: str
) -> OfflineRLConfig:
    if arm == "mlp":
        return OfflineRLConfig(
            model_type=MODEL_ACTOR_CRITIC,
            bc_init=bc_init,
            epochs=epochs,
            beta=beta,
            device=device,
            seed=seed,
        )
    if arm == "et_prior":
        return OfflineRLConfig(
            model_type=MODEL_ENTITY_TRANSFORMER,
            id_embed=True,
            id_embed_init="prior",
            bc_init=None,  # no compatible warm-start from the MLP BC; train fresh
            epochs=epochs,
            beta=beta,
            device=device,
            seed=seed,
        )
    raise ValueError(f"unknown arm {arm!r}")


async def _winrate_vs_heuristic(ckpt: Path, fmt: str, eval_n: int) -> float:
    p1 = build_player(
        "offrl", fmt, checkpoint_path=str(ckpt), max_concurrent_battles=_EVAL_CONCURRENCY
    )
    p2 = build_player("heuristic", fmt, max_concurrent_battles=_EVAL_CONCURRENCY)
    return await evaluate_built(p1, p2, eval_n)


async def run_gate(
    data: str,
    out: str,
    bc_init: str,
    seeds: list[int],
    arms: list[str],
    epochs: int,
    beta: float,
    eval_n: int,
    device: str,
    fmt: str,
) -> dict:
    if "mlp" in arms and not Path(bc_init).exists():
        raise SystemExit(f"BC warm-start '{bc_init}' not found; train it first (lategame train).")
    print(f"shard {data} | arms {arms} | seeds {seeds} | epochs {epochs} | eval_n {eval_n}")

    records: list[dict] = []
    for arm in arms:
        for seed in seeds:
            cfg = _arm_config(arm, seed, epochs, beta, device, bc_init)
            ckpt = _CKPT_DIR / f"offrl_scale_{arm}_s{seed}.pt"
            t0 = time.time()
            metrics = train_offline_rl(data, ckpt, cfg)
            train_s = time.time() - t0
            win_rate = await _winrate_vs_heuristic(ckpt, fmt, eval_n)
            records.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "checkpoint": str(ckpt),
                    "win_rate": win_rate,
                    "val_acc": float(metrics.get("acc", 0.0)),
                    "val_value_mae": float(metrics.get("value_mae", 0.0)),
                    "train_seconds": round(train_s, 1),
                }
            )
            print(
                f"arm={arm:>8} seed={seed} win_rate={win_rate:.3f} "
                f"val_acc={metrics.get('acc', 0.0):.3f} ({train_s:.0f}s train + eval)"
            )

    summary = _summarize(records)
    result = {
        "data": data,
        "bc_init": bc_init,
        "arms": arms,
        "seeds": seeds,
        "epochs": epochs,
        "beta": beta,
        "eval_n": eval_n,
        "opponent": "heuristic",
        "records": records,
        "summary": summary,
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print_table(summary)
    print(f"\nwrote {out_path}")
    return result


def _summarize(records: list[dict]) -> dict:
    by_arm: dict[str, list[float]] = {}
    for rec in records:
        by_arm.setdefault(rec["arm"], []).append(rec["win_rate"])
    return {
        arm: {
            "win_rates": wrs,
            "mean": statistics.mean(wrs),
            "std": statistics.pstdev(wrs) if len(wrs) > 1 else 0.0,
            "min": min(wrs),
            "max": max(wrs),
        }
        for arm, wrs in by_arm.items()
    }


def _print_table(summary: dict) -> None:
    print(f"\n{'arm':>10}  {'win-rate mean+-std':>20}  {'min':>6}  {'max':>6}  (plateau ~27-34%)")
    print("-" * 64)
    for arm, s in summary.items():
        print(
            f"{arm:>10}  {s['mean']:.3f}+-{s['std']:.3f}            "
            f"{s['min']:.3f}  {s['max']:.3f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="offrl_scale_gate")
    parser.add_argument("--data", default="data/resim_v3_gen9rb_rl.npz", help="RL shard (.npz)")
    parser.add_argument("--out", default="results/offrl_scale_gate.json")
    parser.add_argument(
        "--bc-init", default="checkpoints/bc_mlp_resim_v3.pt", help="MLP warm-start ckpt"
    )
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    parser.add_argument("--arms", default="mlp,et_prior", help="Comma-separated: mlp,et_prior")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--beta", type=float, default=1.0, help="AWR temperature")
    parser.add_argument("--eval-n", type=int, default=200, help="Battles vs heuristic per seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--format", dest="battle_format", default=DEFAULT_FORMAT)
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    asyncio.run(
        run_gate(
            args.data,
            args.out,
            args.bc_init,
            seeds,
            arms,
            args.epochs,
            args.beta,
            args.eval_n,
            args.device,
            args.battle_format,
        )
    )


if __name__ == "__main__":
    main()

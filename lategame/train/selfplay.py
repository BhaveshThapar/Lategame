"""Self-play improvement loop: fictitious-play league + fixed anchors (plan.md, M4).

This is the lever M2/M3 couldn't pull. Both plateaued at the *demonstrator ceiling*:
every training target came from a fixed pool of weak bots, so AWR could only
re-weight toward actions already in that weak data. Self-play breaks the ceiling --
the policy generates its own (and, as it improves, better-than-teacher) trajectories,
and re-estimating the value head on that shifting data yields real advantages
relative to current behaviour.

Each iteration (starting from the M3 offline-RL checkpoint as iter 0):

1. the current learner plays a *uniform sample of past-iteration checkpoints* (the
   league -- fictitious play, which avoids the cycling/forgetting collapse of always
   chasing the latest opponent) **plus** the fixed ``heuristic``/``simpleheuristics``
   anchors (they currently beat the policy, so they inject stronger-than-current
   behaviour to climb past the ceiling), recording every turn of both sides;
2. we fine-tune the actor-critic for a few epochs on a sliding window of the most
   recent shards (a small replay buffer), warm-started from the previous iteration;
3. we measure win-rate vs the fixed baselines and head-to-head vs iter 0, append it
   to the improvement curve, and add the new checkpoint to the league.

The value support is pinned to the warm-start checkpoint's ``[v_min, v_max]`` so the
carried-over value head stays calibrated across iterations. Torch is imported lazily
(only the training step needs it), matching ``train.bc`` / ``train.offline_rl``.

The deliverable is ``<out_dir>/curve.json`` + a printed per-iteration table: the
continual-improvement curve that is M4's exit criterion (G3).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from lategame.config import DEFAULT_FORMAT
from lategame.data.collect import PlayerSpec, collect_selfplay, concat_rl_shards, save_rl
from lategame.data.reward import RewardWeights
from lategame.eval.arena import build_player, evaluate_built

CurvePoint = dict[str, float]


@dataclass
class SelfPlayConfig:
    init: str = "checkpoints/offrl_gen9randombattle.pt"
    out_dir: str = "checkpoints/selfplay"
    data_dir: str = "data/selfplay"
    battle_format: str = DEFAULT_FORMAT
    iters: int = 8
    games_per_opp: int = 30
    pop_size: int = 2  # league checkpoints sampled per iteration (fictitious play)
    buffer_iters: int = 3  # sliding window of shards to train each step on
    anchors: tuple[str, ...] = ("heuristic", "simpleheuristics")
    eval_baselines: tuple[str, ...] = ("random", "simpleheuristics", "heuristic")
    eval_n: int = 100
    max_concurrent: int = 20  # concurrent battles during collection / eval (server-bound)
    # Per-iteration short fine-tune (not the M3 30-epoch full fit).
    epochs: int = 4
    batch_size: int = 256
    lr: float = 1e-3
    beta: float = 1.0
    value_coef: float = 0.5
    gamma: float = 0.99
    device: str = "auto"
    seed: int = 0
    weights: RewardWeights = field(default_factory=RewardWeights)


def _sample_league(league: list[str], pop_size: int, rng: random.Random) -> list[str]:
    """Uniformly sample up to ``pop_size`` distinct league members (fictitious play)."""
    return rng.sample(league, min(pop_size, len(league)))


def _print_curve_row(point: CurvePoint) -> None:
    cells = [f"iter {point['iter']:>3}"]
    cells += [f"{key} {value:.3f}" for key, value in point.items() if key != "iter"]
    print("  ".join(cells))


def _write_curve(path: Path, curve: list[CurvePoint]) -> None:
    path.write_text(json.dumps(curve, indent=2))


async def _eval_point(iteration: int, ckpt_path: str, config: SelfPlayConfig) -> CurvePoint:
    """Win-rate of the iteration's checkpoint (greedy) vs each baseline + vs iter 0."""
    point: CurvePoint = {"iter": iteration}
    for base in config.eval_baselines:
        learner = build_player(
            "offrl",
            config.battle_format,
            checkpoint_path=ckpt_path,
            sample=False,
            max_concurrent_battles=config.max_concurrent,
        )
        opponent = build_player(
            base, config.battle_format, max_concurrent_battles=config.max_concurrent
        )
        point[f"vs_{base}"] = round(await evaluate_built(learner, opponent, config.eval_n), 4)
    if iteration > 0:
        learner = build_player(
            "offrl",
            config.battle_format,
            checkpoint_path=ckpt_path,
            sample=False,
            max_concurrent_battles=config.max_concurrent,
        )
        iter0 = build_player(
            "offrl",
            config.battle_format,
            checkpoint_path=config.init,
            sample=False,
            max_concurrent_battles=config.max_concurrent,
        )
        point["vs_iter0"] = round(await evaluate_built(learner, iter0, config.eval_n), 4)
    return point


async def run_selfplay(config: SelfPlayConfig) -> list[CurvePoint]:
    """Run the self-play improvement loop; return (and persist) the per-iter curve."""
    import torch

    from lategame.train.offline_rl import OfflineRLConfig, train_offline_rl

    init_path = Path(config.init)
    if not init_path.exists():
        raise FileNotFoundError(
            f"Warm-start checkpoint not found at '{init_path}'. Train M3 first "
            f"(`lategame train-rl`) or pass --init."
        )
    # Pin the value support + bin count to the warm-start checkpoint so the value head
    # carries over consistently as we warm-start AC -> AC each iteration.
    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    v_min, v_max, n_bins = float(ckpt["v_min"]), float(ckpt["v_max"]), int(ckpt["n_bins"])

    out_dir = Path(config.out_dir)
    data_dir = Path(config.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    latest = str(init_path)  # the current learner's checkpoint
    league = [str(init_path)]  # all checkpoints usable as opponents
    shard_paths: list[str] = []

    curve: list[CurvePoint] = [await _eval_point(0, latest, config)]
    _print_curve_row(curve[0])
    _write_curve(out_dir / "curve.json", curve)

    for k in range(1, config.iters + 1):
        opponents = [
            PlayerSpec("offrl", checkpoint_path=p, sample=True)
            for p in _sample_league(league, config.pop_size, rng)
        ]
        opponents += [PlayerSpec(a) for a in config.anchors]
        learner = PlayerSpec("offrl", checkpoint_path=latest, sample=True)

        dataset = await collect_selfplay(
            learner,
            opponents,
            config.games_per_opp,
            config.battle_format,
            config.weights,
            config.gamma,
            max_concurrent=config.max_concurrent,
        )
        shard_path = str(data_dir / f"iter_{k:02d}.npz")
        save_rl(dataset, shard_path)
        shard_paths.append(shard_path)

        buffer_path = str(data_dir / f"buffer_{k:02d}.npz")
        concat_rl_shards(shard_paths[-config.buffer_iters :], buffer_path)

        ckpt_path = str(out_dir / f"iter_{k:02d}.pt")
        train_offline_rl(
            buffer_path,
            ckpt_path,
            OfflineRLConfig(
                epochs=config.epochs,
                batch_size=config.batch_size,
                lr=config.lr,
                n_bins=n_bins,
                beta=config.beta,
                value_coef=config.value_coef,
                device=config.device,
                bc_init=latest,
                v_min=v_min,
                v_max=v_max,
                seed=config.seed,
            ),
        )
        latest = ckpt_path
        league.append(ckpt_path)

        point = await _eval_point(k, latest, config)
        curve.append(point)
        _print_curve_row(point)
        _write_curve(out_dir / "curve.json", curve)

    print(f"wrote curve ({len(curve)} points) to {out_dir / 'curve.json'}")
    return curve

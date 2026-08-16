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

from poke_env.teambuilder.teambuilder import Teambuilder

from lategame.config import DEFAULT_FORMAT
from lategame.data.collect import PlayerSpec, collect_selfplay, concat_rl_shards, save_rl
from lategame.data.reward import RewardWeights
from lategame.eval.arena import build_player, evaluate_built, policy_agent
from lategame.features.codec import codec_for
from lategame.teambuilding.pool import TeamPool

CurvePoint = dict[str, float]


@dataclass
class SelfPlayConfig:
    init: str = "checkpoints/offrl_gen9randombattle.pt"
    out_dir: str = "checkpoints/selfplay"
    data_dir: str = "data/selfplay"
    battle_format: str = DEFAULT_FORMAT
    team_pool: str | None = None  # packed-team pool; REQUIRED on a teambuilt format (gen9ou, VGC)
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
    # B6f's backstop, reachable at last: `collect_selfplay` has accepted this since the VGC loop
    # was found, and nothing on the M4 path could set it. None is exact identity.
    max_battle_turns: int | None = None


def _sample_league(league: list[str], pop_size: int, rng: random.Random) -> list[str]:
    """Uniformly sample up to ``pop_size`` distinct league members (fictitious play)."""
    return rng.sample(league, min(pop_size, len(league)))


def _print_curve_row(point: CurvePoint) -> None:
    cells = [f"iter {point['iter']:>3}"]
    cells += [f"{key} {value:.3f}" for key, value in point.items() if key != "iter"]
    print("  ".join(cells))


def _write_curve(path: Path, curve: list[CurvePoint]) -> None:
    path.write_text(json.dumps(curve, indent=2))


async def _eval_point(
    iteration: int,
    ckpt_path: str,
    config: SelfPlayConfig,
    team: str | Teambuilder | None = None,
) -> CurvePoint:
    """Win-rate of the iteration's checkpoint (greedy) vs each baseline + vs iter 0.

    The greedy agent is ASKED for, not named: ``build_player`` refuses ``offrl`` on a doubles
    format, so hardcoding it made this the first thing a VGC self-play run hit -- a ValueError
    from iteration 0's eval point, before a battle was ever requested.

    ``team`` reaches every player including the fixed baselines, which have no teams of their
    own; on a teambuilt format a baseline without one cannot start a battle. The turn cap goes
    to the learner builds only, matching ``train.ppo._eval_point``.
    """
    point: CurvePoint = {"iter": iteration}
    learned = policy_agent(config.battle_format)
    for base in config.eval_baselines:
        learner = build_player(
            learned,
            config.battle_format,
            checkpoint_path=ckpt_path,
            sample=False,
            max_concurrent_battles=config.max_concurrent,
            team=team,
            max_battle_turns=config.max_battle_turns,
        )
        opponent = build_player(
            base, config.battle_format, max_concurrent_battles=config.max_concurrent, team=team
        )
        point[f"vs_{base}"] = round(await evaluate_built(learner, opponent, config.eval_n), 4)
    if iteration > 0:
        learner = build_player(
            learned,
            config.battle_format,
            checkpoint_path=ckpt_path,
            sample=False,
            max_concurrent_battles=config.max_concurrent,
            team=team,
            max_battle_turns=config.max_battle_turns,
        )
        iter0 = build_player(
            learned,
            config.battle_format,
            checkpoint_path=config.init,
            sample=False,
            max_concurrent_battles=config.max_concurrent,
            team=team,
            max_battle_turns=config.max_battle_turns,
        )
        point["vs_iter0"] = round(await evaluate_built(learner, iter0, config.eval_n), 4)
    return point


def _check_warm_start(ckpt: dict, config: SelfPlayConfig) -> None:
    """Refuse an init the loop cannot honour, before the first battle.

    Mirrors ``train.ppo.run_ppo``'s first two checks. M4 had neither: a singles warm start on a
    doubles format got as far as building players, and the mismatch surfaced as a shape error
    inside the fine-tune step -- after a full iteration of collection had already been paid for.

    ``run_ppo``'s THIRD check, which refuses a singles-tuned ``loop_penalty`` on doubles, has no
    counterpart here on purpose: ``SelfPlayConfig`` carries no ``loop_penalty``. Its only honest
    doubles value would be 0.0, since ``DoublesLoopGuard`` is a per-slot penalty over a different
    action layout, so the field would be a knob with one correct setting.

    The value-support check is a THIRD one this function did not originally have, and its absence
    reproduced the exact failure the other two exist to prevent. ``doubles_bc_vgc_v2.pt``
    clears the format and encoder checks and then dies one line into ``run_selfplay`` on
    ``KeyError: 'v_min'``: a BC checkpoint stamps ``n_bins`` but no value support, because BC never
    fits a critic. A bare KeyError one statement past a guard whose whole job is naming the refusal
    is the same defect the guard was written for.
    """
    # Checked FIRST: the encoder check below is only meaningful once we know which format's codec
    # to check against. A checkpoint with no `battle_format` key is a legacy RB warm start and
    # passes, which is what keeps every pre-B6f arm runnable.
    battle_format = str(ckpt.get("battle_format", config.battle_format))
    if battle_format != config.battle_format:
        raise ValueError(
            f"format mismatch: init checkpoint is '{battle_format}' but config is "
            f"'{config.battle_format}'; collection/eval would differ -- pass --format "
            f"{battle_format}"
        )
    codec = codec_for(battle_format)
    if (
        ckpt.get("obs_version") != codec.obs_version
        or ckpt.get("input_dim") != codec.obs_dim
        or ckpt.get("n_actions") != codec.n_actions
    ):
        raise ValueError(
            f"init checkpoint encoder mismatch (ckpt {ckpt.get('obs_version')}/"
            f"{ckpt.get('input_dim')}/{ckpt.get('n_actions')} vs {codec.name} codec "
            f"{codec.obs_version}/{codec.obs_dim}/{codec.n_actions}); cannot warm-start "
            f"self-play. Retrain."
        )
    missing = [k for k in ("v_min", "v_max", "n_bins") if ckpt.get(k) is None]
    if missing:
        raise ValueError(
            f"init checkpoint carries no value support ({', '.join(missing)} missing or None); "
            f"self-play warm-starts AC -> AC each iteration and pins the critic's support to the "
            f"init, so there is nothing to pin to. This is what a BC checkpoint looks like -- "
            f"warm-start from the offline-RL one trained on top of it instead."
        )


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
    _check_warm_start(ckpt, config)
    v_min, v_max, n_bins = float(ckpt["v_min"]), float(ckpt["v_max"]), int(ckpt["n_bins"])

    # One pool for eval (poke-env re-draws per battle, so the players sharing it is fine);
    # collection builds its own two, seeded apart, inside `collect_selfplay`.
    team = TeamPool.from_packed_file(config.team_pool) if config.team_pool else None

    out_dir = Path(config.out_dir)
    data_dir = Path(config.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    latest = str(init_path)  # the current learner's checkpoint
    league = [str(init_path)]  # all checkpoints usable as opponents
    shard_paths: list[str] = []

    curve: list[CurvePoint] = [await _eval_point(0, latest, config, team=team)]
    _print_curve_row(curve[0])
    _write_curve(out_dir / "curve.json", curve)

    # The league name, asked for once. `policy_agent`, not `rollout_agent`: M4 records through
    # `_build_recording_player`'s codec-driven mixin, not PPO's log-prob rollout path.
    league_agent = policy_agent(config.battle_format)
    for k in range(1, config.iters + 1):
        opponents = [
            PlayerSpec(league_agent, checkpoint_path=p, sample=True)
            for p in _sample_league(league, config.pop_size, rng)
        ]
        opponents += [PlayerSpec(a) for a in config.anchors]
        learner = PlayerSpec(league_agent, checkpoint_path=latest, sample=True)

        dataset = await collect_selfplay(
            learner,
            opponents,
            config.games_per_opp,
            config.battle_format,
            config.weights,
            config.gamma,
            max_concurrent=config.max_concurrent,
            max_battle_turns=config.max_battle_turns,
            team_pool=config.team_pool,
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

        point = await _eval_point(k, latest, config, team=team)
        curve.append(point)
        _print_curve_row(point)
        _write_curve(out_dir / "curve.json", curve)

    print(f"wrote curve ({len(curve)} points) to {out_dir / 'curve.json'}")
    return curve

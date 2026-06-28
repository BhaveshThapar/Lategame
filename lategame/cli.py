"""R-CTRL seed: command-line control surface.

    python -m lategame.cli evaluate --p1 heuristic --p2 random --n 100
    python -m lategame.cli play     --p1 heuristic --p2 random
    python -m lategame.cli collect  --pool random,maxbasepower,simpleheuristics,heuristic --n 50
    python -m lategame.cli train    --data data/gen9rb.npz --epochs 20
    python -m lategame.cli collect-rl --pool random,maxbasepower,heuristic --n 50
    python -m lategame.cli train-rl --data data/gen9rb_rl.npz --bc-init checkpoints/bc.pt
    python -m lategame.cli selfplay --init checkpoints/offrl_gen9randombattle.pt --iters 8
    python -m lategame.cli ppo      --init checkpoints/offrl_gen9randombattle.pt --iters 8
    python -m lategame.cli fetch-replays --min-rating 1200 --limit 200
    python -m lategame.cli resim-replays --out data/resim_gen9rb_rl.npz

``collect``/``train`` are the M2 BC pipeline; ``collect-rl``/``train-rl`` are the
M3 offline-RL pipeline; ``selfplay`` is the M4 self-play improvement loop; ``ppo`` is
the M5 on-policy PPO self-play loop. ``fetch-replays`` + ``ingest-replays`` are the M6
human-replay pipeline (public-log POV); ``resim-replays`` is M6 v2, re-simulating each
replay's inputlog for a full-fidelity (request-based) POV. Torch is imported lazily inside
the train handlers so the eval/collect/data commands keep running without ``[ml]``.
"""

from __future__ import annotations

import argparse
import asyncio

from lategame.config import DEFAULT_FORMAT
from lategame.eval.arena import AGENTS, EvalResult, evaluate

_DEFAULT_POOL = "random,maxbasepower,simpleheuristics,heuristic"
_DEFAULT_BC_INIT = "checkpoints/bc_gen9randombattle.pt"
_MODEL_CHOICES = ["actor_critic", "entity_transformer"]


def _print_result(result: EvalResult) -> None:
    pct = 100.0 * result.p1_win_rate
    print(
        f"[{result.battle_format}] {result.p1_name} vs {result.p2_name} "
        f"over {result.n_battles} battle(s): "
        f"{result.p1_name} win rate = {pct:.1f}%"
    )


async def _run_eval(args: argparse.Namespace) -> None:
    for who in (args.p1, args.p2):
        if who not in AGENTS:
            raise SystemExit(f"Unknown agent '{who}'. Choose from: {', '.join(AGENTS)}")
    result = await evaluate(args.p1, args.p2, args.n, args.battle_format)
    _print_result(result)


async def _run_collect(args: argparse.Namespace) -> None:
    from lategame.data.collect import collect, save

    pool = [name.strip() for name in args.pool.split(",") if name.strip()]
    dataset = await collect(pool, args.n, args.battle_format)
    save(dataset, args.out)


def _run_train(args: argparse.Namespace) -> None:
    from lategame.train.bc import TrainConfig, train_bc

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        device=args.device,
        model_type=args.model_type,
        id_embed=args.id_embed,
        id_embed_init=args.id_embed_init,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    train_bc(args.data, args.out, config)


async def _run_collect_rl(args: argparse.Namespace) -> None:
    from lategame.data.collect import collect_trajectories, save_rl
    from lategame.data.reward import RewardWeights

    pool = [name.strip() for name in args.pool.split(",") if name.strip()]
    weights = RewardWeights(
        hp_value=args.hp_value,
        fainted_value=args.fainted_value,
        status_value=args.status_value,
        victory_value=args.victory_value,
    )
    dataset = await collect_trajectories(pool, args.n, args.battle_format, weights, args.gamma)
    save_rl(dataset, args.out)


def _run_train_rl(args: argparse.Namespace) -> None:
    from lategame.train.offline_rl import OfflineRLConfig, train_offline_rl

    # The MLP BC checkpoint cannot warm-start a transformer (no compatible trunk);
    # drop the default BC init for non-MLP models so the transformer trains fresh.
    bc_init = args.bc_init or None
    if args.model != "actor_critic" and bc_init == _DEFAULT_BC_INIT:
        bc_init = None

    config = OfflineRLConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        n_bins=args.n_bins,
        beta=args.beta,
        value_coef=args.value_coef,
        device=args.device,
        bc_init=bc_init,
        model_type=args.model,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )
    train_offline_rl(args.data, args.out, config)


async def _run_selfplay(args: argparse.Namespace) -> None:
    from lategame.train.selfplay import SelfPlayConfig, run_selfplay

    config = SelfPlayConfig(
        init=args.init,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        battle_format=args.battle_format,
        iters=args.iters,
        games_per_opp=args.games_per_opp,
        pop_size=args.pop_size,
        buffer_iters=args.buffer_iters,
        anchors=tuple(a.strip() for a in args.anchors.split(",") if a.strip()),
        eval_baselines=tuple(b.strip() for b in args.eval_baselines.split(",") if b.strip()),
        eval_n=args.eval_n,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        value_coef=args.value_coef,
        gamma=args.gamma,
        device=args.device,
        seed=args.seed,
    )
    await run_selfplay(config)


async def _run_ppo(args: argparse.Namespace) -> None:
    from lategame.train.ppo import PPOConfig, run_ppo

    config = PPOConfig(
        init=args.init,
        out_dir=args.out_dir,
        battle_format=args.battle_format,
        iters=args.iters,
        games_per_opp=args.games_per_opp,
        pop_size=args.pop_size,
        max_concurrent=args.max_concurrent,
        anchors=tuple(a.strip() for a in args.anchors.split(",") if a.strip()),
        eval_baselines=tuple(b.strip() for b in args.eval_baselines.split(",") if b.strip()),
        eval_n=args.eval_n,
        epochs=args.epochs,
        minibatch=args.minibatch,
        lr=args.lr,
        clip_eps=args.clip_eps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        device=args.device,
        seed=args.seed,
    )
    await run_ppo(config)


def _run_fetch_replays(args: argparse.Namespace) -> None:
    from lategame.data.replays import fetch_replays

    fetch_replays(
        battle_format=args.battle_format,
        min_rating=args.min_rating,
        limit=args.limit,
        cache_dir=args.cache_dir,
        max_pages=args.max_pages,
        sleep=args.sleep,
        require_rating=not args.allow_unrated,
    )


def _run_ingest_replays(args: argparse.Namespace) -> None:
    from lategame.data.ingest import ingest_and_save
    from lategame.data.replays import cached_replay_paths
    from lategame.data.reward import RewardWeights

    paths = cached_replay_paths(args.cache_dir)
    if not paths:
        raise SystemExit(f"No cached replays under '{args.cache_dir}'. Run `fetch-replays` first.")
    weights = RewardWeights(
        hp_value=args.hp_value,
        fainted_value=args.fainted_value,
        status_value=args.status_value,
        victory_value=args.victory_value,
    )
    ingest_and_save(
        paths,
        rl_out=args.out,
        bc_out=args.bc_out,
        weights=weights,
        gamma=args.gamma,
        battle_format=args.battle_format,
    )


def _run_resim_replays(args: argparse.Namespace) -> None:
    from lategame.data.replays import cached_replay_paths
    from lategame.data.resim import resim_and_save
    from lategame.data.reward import RewardWeights

    paths = cached_replay_paths(args.cache_dir)
    if not paths:
        raise SystemExit(f"No cached replays under '{args.cache_dir}'. Run `fetch-replays` first.")
    weights = RewardWeights(
        hp_value=args.hp_value,
        fainted_value=args.fainted_value,
        status_value=args.status_value,
        victory_value=args.victory_value,
    )
    resim_and_save(
        paths,
        rl_out=args.out,
        bc_out=args.bc_out,
        weights=weights,
        gamma=args.gamma,
        battle_format=args.battle_format,
        showdown_dir=args.showdown_dir,
        node=args.node,
    )


def _add_eval_args(parser: argparse.ArgumentParser, default_n: int) -> None:
    agents = ", ".join(AGENTS)
    parser.add_argument("--p1", required=True, help=f"Agent for player 1 ({agents})")
    parser.add_argument("--p2", required=True, help=f"Agent for player 2 ({agents})")
    parser.add_argument("--n", type=int, default=default_n, help="Number of battles")
    parser.add_argument(
        "--format", dest="battle_format", default=DEFAULT_FORMAT, help="Showdown format string"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lategame")
    sub = parser.add_subparsers(dest="command", required=True)

    _add_eval_args(sub.add_parser("evaluate", help="Run N battles and report win rate"), 100)
    _add_eval_args(sub.add_parser("play", help="Play a single battle and report the winner"), 1)

    collect = sub.add_parser("collect", help="Generate a reward-filtered self-play BC dataset")
    collect.add_argument("--pool", default=_DEFAULT_POOL, help="Comma-separated demonstrators")
    collect.add_argument("--n", type=int, default=50, help="Battles per agent pair")
    collect.add_argument("--out", default="data/gen9rb.npz", help="Output .npz path")
    collect.add_argument(
        "--format", dest="battle_format", default=DEFAULT_FORMAT, help="Showdown format string"
    )

    train = sub.add_parser("train", help="Train the BC policy on a collected dataset")
    train.add_argument("--data", default="data/gen9rb.npz", help="Dataset .npz path")
    train.add_argument(
        "--out", default="checkpoints/bc_gen9randombattle.pt", help="Checkpoint path"
    )
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    train.add_argument(
        "--model-type",
        default="bc_policy",
        choices=["bc_policy", "entity_transformer", "actor_critic"],
        help="Architecture: flat MLP (default) or factory model",
    )
    train.add_argument(
        "--no-id-embed",
        dest="id_embed",
        action="store_false",
        help="entity_transformer: disable learned species/move/item/ability embeddings (ablation)",
    )
    train.set_defaults(id_embed=True)
    train.add_argument(
        "--id-embed-init",
        default="random",
        choices=["random", "prior"],
        help="entity_transformer: 'prior' warm-starts species/move embeddings from dex features",
    )
    train.add_argument("--seed", type=int, default=0, help="Init + train/val-split seed")
    train.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Train on a deterministic nested subset of N samples (data-scaling sweep)",
    )

    collect_rl = sub.add_parser(
        "collect-rl", help="Generate an offline-RL dataset (all turns + shaped rewards)"
    )
    collect_rl.add_argument("--pool", default=_DEFAULT_POOL, help="Comma-separated demonstrators")
    collect_rl.add_argument("--n", type=int, default=50, help="Battles per agent pair")
    collect_rl.add_argument("--out", default="data/gen9rb_rl.npz", help="Output .npz path")
    collect_rl.add_argument(
        "--format", dest="battle_format", default=DEFAULT_FORMAT, help="Showdown format string"
    )
    collect_rl.add_argument("--gamma", type=float, default=0.99, help="Return discount")
    collect_rl.add_argument("--hp-value", type=float, default=0.3, help="Shaping: HP weight")
    collect_rl.add_argument("--fainted-value", type=float, default=1.0, help="Shaping: faint")
    collect_rl.add_argument("--status-value", type=float, default=0.1, help="Shaping: status")
    collect_rl.add_argument("--victory-value", type=float, default=1.0, help="Terminal win/loss")

    train_rl = sub.add_parser("train-rl", help="Train the offline-RL (value+AWR) policy")
    train_rl.add_argument("--data", default="data/gen9rb_rl.npz", help="Dataset .npz path")
    train_rl.add_argument(
        "--out", default="checkpoints/offrl_gen9randombattle.pt", help="Checkpoint path"
    )
    train_rl.add_argument(
        "--bc-init",
        default=_DEFAULT_BC_INIT,
        help="BC checkpoint to warm-start from (pass '' to train from scratch)",
    )
    train_rl.add_argument(
        "--model",
        default="actor_critic",
        choices=_MODEL_CHOICES,
        help="Architecture: flat MLP (actor_critic) or entity transformer",
    )
    train_rl.add_argument("--epochs", type=int, default=30)
    train_rl.add_argument("--batch-size", type=int, default=256)
    train_rl.add_argument("--lr", type=float, default=1e-3)
    train_rl.add_argument("--hidden-dim", type=int, default=256, help="MLP trunk width")
    train_rl.add_argument("--d-model", type=int, default=128, help="Transformer token width")
    train_rl.add_argument("--n-layers", type=int, default=2, help="Transformer encoder layers")
    train_rl.add_argument("--n-heads", type=int, default=4, help="Transformer attention heads")
    train_rl.add_argument("--n-bins", type=int, default=51)
    train_rl.add_argument("--beta", type=float, default=1.0, help="AWR temperature")
    train_rl.add_argument("--value-coef", type=float, default=0.5, help="Value-loss weight")
    train_rl.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])

    selfplay = sub.add_parser("selfplay", help="Run the M4 self-play improvement loop")
    selfplay.add_argument(
        "--init",
        default="checkpoints/offrl_gen9randombattle.pt",
        help="Warm-start (M3 offline-RL) checkpoint = iteration 0",
    )
    selfplay.add_argument("--out-dir", default="checkpoints/selfplay", help="Checkpoints + curve")
    selfplay.add_argument("--data-dir", default="data/selfplay", help="Per-iteration shards")
    selfplay.add_argument("--iters", type=int, default=8, help="Self-play iterations")
    selfplay.add_argument("--games-per-opp", type=int, default=30, help="Battles per opponent")
    selfplay.add_argument("--pop-size", type=int, default=2, help="League members sampled per iter")
    selfplay.add_argument("--buffer-iters", type=int, default=3, help="Sliding shard window")
    selfplay.add_argument(
        "--anchors", default="heuristic,simpleheuristics", help="Fixed anchor opponents"
    )
    selfplay.add_argument(
        "--eval-baselines",
        default="random,simpleheuristics,heuristic",
        help="Baselines to chart win-rate against each iteration",
    )
    selfplay.add_argument("--eval-n", type=int, default=100, help="Battles per eval matchup")
    selfplay.add_argument("--epochs", type=int, default=4, help="Fine-tune epochs per iteration")
    selfplay.add_argument("--batch-size", type=int, default=256)
    selfplay.add_argument("--lr", type=float, default=1e-3)
    selfplay.add_argument("--beta", type=float, default=1.0, help="AWR temperature")
    selfplay.add_argument("--value-coef", type=float, default=0.5, help="Value-loss weight")
    selfplay.add_argument("--gamma", type=float, default=0.99, help="Return discount")
    selfplay.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    selfplay.add_argument("--seed", type=int, default=0)
    selfplay.add_argument(
        "--format", dest="battle_format", default=DEFAULT_FORMAT, help="Showdown format string"
    )

    ppo = sub.add_parser("ppo", help="Run the M5 on-policy PPO self-play loop")
    ppo.add_argument(
        "--init",
        default="checkpoints/offrl_gen9randombattle.pt",
        help="Warm-start (M3 offline-RL / M4 self-play) checkpoint = iteration 0",
    )
    ppo.add_argument("--out-dir", default="checkpoints/ppo", help="Checkpoints + curve")
    ppo.add_argument("--iters", type=int, default=8, help="PPO iterations")
    ppo.add_argument("--games-per-opp", type=int, default=8, help="Rollout battles per opponent")
    ppo.add_argument("--pop-size", type=int, default=2, help="League members sampled per iter")
    ppo.add_argument(
        "--max-concurrent", type=int, default=20, help="Concurrent battles in rollout/eval"
    )
    ppo.add_argument("--anchors", default="simpleheuristics", help="Fixed anchor opponents")
    ppo.add_argument(
        "--eval-baselines",
        default="random,simpleheuristics,heuristic",
        help="Baselines to chart win-rate against each iteration",
    )
    ppo.add_argument("--eval-n", type=int, default=100, help="Battles per eval matchup")
    ppo.add_argument("--epochs", type=int, default=4, help="PPO epochs per rollout")
    ppo.add_argument("--minibatch", type=int, default=256)
    ppo.add_argument("--lr", type=float, default=2.5e-4)
    ppo.add_argument("--clip-eps", type=float, default=0.2, help="PPO surrogate clip")
    ppo.add_argument("--gamma", type=float, default=0.99, help="Return discount")
    ppo.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    ppo.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus weight")
    ppo.add_argument("--value-coef", type=float, default=0.5, help="Value-loss weight")
    ppo.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clip norm")
    ppo.add_argument("--target-kl", type=float, default=0.03, help="Approx-KL early-stop target")
    ppo.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ppo.add_argument("--seed", type=int, default=0)
    ppo.add_argument(
        "--format", dest="battle_format", default=DEFAULT_FORMAT, help="Showdown format string"
    )

    fetch = sub.add_parser(
        "fetch-replays", help="Download + cache rated human replays (M6 data plan)"
    )
    fetch.add_argument("--min-rating", type=int, default=1200, help="Keep replays >= this rating")
    fetch.add_argument("--limit", type=int, default=200, help="Number of rated replays to keep")
    fetch.add_argument(
        "--cache-dir", default="replays/gen9randombattle", help="Where to cache replay JSON"
    )
    fetch.add_argument("--sleep", type=float, default=0.4, help="Seconds between network requests")
    fetch.add_argument("--max-pages", type=int, default=80, help="Max search index pages to walk")
    fetch.add_argument(
        "--allow-unrated", action="store_true", help="Also keep replays with unknown rating"
    )
    fetch.add_argument(
        "--format", dest="battle_format", default=DEFAULT_FORMAT, help="Showdown format string"
    )

    ingest = sub.add_parser(
        "ingest-replays", help="Reconstruct cached replays into RL (+ BC) shards"
    )
    ingest.add_argument(
        "--cache-dir", default="replays/gen9randombattle", help="Cached replay JSON dir"
    )
    ingest.add_argument(
        "--out", default="data/replays_gen9rb_rl.npz", help="Offline-RL shard output (.npz)"
    )
    ingest.add_argument(
        "--bc-out", default="data/replays_gen9rb_bc.npz", help="Winners-only BC shard (.npz)"
    )
    ingest.add_argument("--gamma", type=float, default=0.99, help="Return discount")
    ingest.add_argument("--hp-value", type=float, default=0.3, help="Shaping: HP weight")
    ingest.add_argument("--fainted-value", type=float, default=1.0, help="Shaping: faint")
    ingest.add_argument("--status-value", type=float, default=0.1, help="Shaping: status")
    ingest.add_argument("--victory-value", type=float, default=1.0, help="Terminal win/loss")
    ingest.add_argument(
        "--format", dest="battle_format", default=DEFAULT_FORMAT, help="Showdown format string"
    )

    resim = sub.add_parser(
        "resim-replays",
        help="Re-simulate cached replays for full-fidelity POV shards (M6 v2)",
    )
    resim.add_argument(
        "--cache-dir", default="replays/gen9randombattle", help="Cached replay JSON dir"
    )
    resim.add_argument(
        "--out", default="data/resim_gen9rb_rl.npz", help="Offline-RL shard output (.npz)"
    )
    resim.add_argument(
        "--bc-out", default="data/resim_gen9rb_bc.npz", help="Winners-only BC shard (.npz)"
    )
    resim.add_argument(
        "--showdown-dir",
        default="third_party/pokemon-showdown",
        help="Vendored pokemon-showdown dir (must have a built dist/)",
    )
    resim.add_argument("--node", default="node", help="Node executable for the resim driver")
    resim.add_argument("--gamma", type=float, default=0.99, help="Return discount")
    resim.add_argument("--hp-value", type=float, default=0.3, help="Shaping: HP weight")
    resim.add_argument("--fainted-value", type=float, default=1.0, help="Shaping: faint")
    resim.add_argument("--status-value", type=float, default=0.1, help="Shaping: status")
    resim.add_argument("--victory-value", type=float, default=1.0, help="Terminal win/loss")
    resim.add_argument(
        "--format", dest="battle_format", default=DEFAULT_FORMAT, help="Showdown format string"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command in ("evaluate", "play"):
        asyncio.run(_run_eval(args))
    elif args.command == "collect":
        asyncio.run(_run_collect(args))
    elif args.command == "train":
        _run_train(args)
    elif args.command == "collect-rl":
        asyncio.run(_run_collect_rl(args))
    elif args.command == "train-rl":
        _run_train_rl(args)
    elif args.command == "selfplay":
        asyncio.run(_run_selfplay(args))
    elif args.command == "ppo":
        asyncio.run(_run_ppo(args))
    elif args.command == "fetch-replays":
        _run_fetch_replays(args)
    elif args.command == "ingest-replays":
        _run_ingest_replays(args)
    elif args.command == "resim-replays":
        _run_resim_replays(args)


if __name__ == "__main__":
    main()

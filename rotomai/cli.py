"""R-CTRL seed: command-line control surface.

    python -m rotomai.cli evaluate --p1 heuristic --p2 random --n 100
    python -m rotomai.cli play     --p1 heuristic --p2 random
    python -m rotomai.cli collect  --pool random,maxbasepower,simpleheuristics,heuristic --n 50
    python -m rotomai.cli train    --data data/gen9rb.npz --epochs 20
    python -m rotomai.cli collect-rl --pool random,maxbasepower,heuristic --n 50
    python -m rotomai.cli train-rl --data data/gen9rb_rl.npz --bc-init checkpoints/bc.pt
    python -m rotomai.cli selfplay --init checkpoints/offrl_gen9randombattle.pt --iters 8
    python -m rotomai.cli ppo      --init checkpoints/offrl_gen9randombattle.pt --iters 8
    python -m rotomai.cli live     --agent ppo --mode accept --n 5
    python -m rotomai.cli eval-ladder --format gen9ou --n 150
    python -m rotomai.cli fetch-replays --min-rating 1200 --limit 200
    python -m rotomai.cli resim-replays --out data/resim_gen9rb_rl.npz

``collect``/``train`` are the M2 BC pipeline; ``collect-rl``/``train-rl`` are the
M3 offline-RL pipeline; ``selfplay`` is the M4 self-play improvement loop; ``ppo`` is
the M5 on-policy PPO self-play loop; ``live`` plays a LIVE server (M5 deploy / G1 -- see
``rotomai/live/`` and plan.md 15 before using ``--mode ladder``). ``evaluate`` scores one agent
against ONE fixed baseline, where win rate is the sufficient statistic; ``eval-ladder`` rates a
whole FIELD against itself, which is the only condition under which GXE/Glicko-1 say anything a
win rate did not (see ``rotomai/eval/ladder.py``). ``fetch-replays`` +
``ingest-replays`` are the M6
human-replay pipeline (public-log POV); ``resim-replays`` is M6 v2, re-simulating each
replay's inputlog for a full-fidelity (request-based) POV. Torch is imported lazily inside
the train handlers so the eval/collect/data commands keep running without ``[ml]``.
"""

from __future__ import annotations

import argparse
import asyncio

# NOTE: `live.policy` has NO imports of its own, which is why it -- and not `live.session` --
# is the module imported here: building --help must not pay for poke-env or torch.
from rotomai.config import DEFAULT_FORMAT
from rotomai.eval.arena import AGENTS, EvalResult, evaluate

# `eval.ladder` and `eval.rating` are pure-Python (no poke-env, no torch), so importing the
# constants that fill in --help text here costs nothing.
from rotomai.eval.ladder import DEFAULT_ANCHOR, DEFAULT_FIELD, DEFAULT_LOOP_PENALTY
from rotomai.eval.rating import MAX_RD
from rotomai.live.policy import ALLOW_LADDER_ENV, LADDER_ACK, PASSWORD_ENV, USERNAME_ENV

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


def _default_team_pool(battle_format: str, explicit: str | None) -> str | None:
    """The committed pool for ``battle_format``, unless the caller named one.

    `build_player` refuses a teamless build on a teambuilt format, which made `rotomai evaluate
    --format gen9ou` unrunnable -- and before that refusal it was silently WORSE, since Showdown
    rejected every challenge with a popup and the command reported a 0.000 win rate. Both pools are
    committed, so there is one sensible answer per format and no reason to make every invocation
    spell it out.
    """
    from rotomai.config import is_doubles_format
    from rotomai.teambuilding.pool import DEFAULT_OU_POOL, DEFAULT_VGC_POOL

    if explicit:
        return explicit
    if "randombattle" in battle_format:
        return None  # the server supplies the teams; passing a pool would be wrong
    return str(DEFAULT_VGC_POOL if is_doubles_format(battle_format) else DEFAULT_OU_POOL)


async def _run_eval(args: argparse.Namespace) -> None:
    for who in (args.p1, args.p2):
        if who not in AGENTS:
            raise SystemExit(f"Unknown agent '{who}'. Choose from: {', '.join(AGENTS)}")
    result = await evaluate(
        args.p1, args.p2, args.n, args.battle_format,
        team_pool=_default_team_pool(args.battle_format, args.team_pool),
        p1_checkpoint=args.p1_checkpoint,
        p2_checkpoint=args.p2_checkpoint,
    )
    _print_result(result)


async def _run_collect(args: argparse.Namespace) -> None:
    from rotomai.data.collect import collect, save

    pool = [name.strip() for name in args.pool.split(",") if name.strip()]
    dataset = await collect(pool, args.n, args.battle_format)
    save(dataset, args.out)


def _run_train(args: argparse.Namespace) -> None:
    from rotomai.train.bc import TrainConfig, train_bc

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
        pp_aug_frac=args.pp_aug_frac,
        pp_aug_turn_threshold=args.pp_aug_turn_threshold,
        pp_noise_std=args.pp_noise_std,
        pp_resample_frac=args.pp_resample_frac,
    )
    train_bc(args.data, args.out, config)


async def _run_collect_rl(args: argparse.Namespace) -> None:
    from rotomai.data.collect import collect_trajectories, save_rl
    from rotomai.data.reward import RewardWeights

    pool = [name.strip() for name in args.pool.split(",") if name.strip()]
    weights = RewardWeights(
        hp_value=args.hp_value,
        fainted_value=args.fainted_value,
        status_value=args.status_value,
        victory_value=args.victory_value,
    )
    dataset = await collect_trajectories(
        pool,
        args.n,
        args.battle_format,
        weights,
        args.gamma,
        team_pool=args.team_pool,
        max_battle_turns=args.max_battle_turns,
    )
    save_rl(dataset, args.out)


def _run_train_rl(args: argparse.Namespace) -> None:
    from rotomai.train.offline_rl import OfflineRLConfig, train_offline_rl

    # The MLP BC checkpoint cannot warm-start a transformer (no compatible trunk);
    # drop the default BC init for non-MLP models so the transformer trains fresh.
    bc_init = args.bc_init or None
    if args.model != "actor_critic" and bc_init == _DEFAULT_BC_INIT:
        bc_init = None

    # Unset arch flags fall back to the dataclass defaults; passing any of them marks the
    # request explicit, so a conflicting warm-start raises instead of silently winning.
    defaults = OfflineRLConfig()
    requested = {k: getattr(args, k) for k in ("d_model", "n_layers", "n_heads", "ff_dim")}
    arch = {k: getattr(defaults, k) if v is None else v for k, v in requested.items()}

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
        **arch,
        arch_explicit=any(v is not None for v in requested.values()),
        id_embed=args.id_embed,
        id_embed_init=args.id_embed_init,
        seed=args.seed,
    )
    train_offline_rl(args.data, args.out, config)


async def _run_selfplay(args: argparse.Namespace) -> None:
    from rotomai.train.selfplay import SelfPlayConfig, run_selfplay

    config = SelfPlayConfig(
        init=args.init,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        battle_format=args.battle_format,
        team_pool=args.team_pool,
        max_battle_turns=args.max_battle_turns,
        iters=args.iters,
        games_per_opp=args.games_per_opp,
        pop_size=args.pop_size,
        buffer_iters=args.buffer_iters,
        anchors=tuple(a.strip() for a in args.anchors.split(",") if a.strip()),
        eval_baselines=tuple(b.strip() for b in args.eval_baselines.split(",") if b.strip()),
        eval_n=args.eval_n,
        max_concurrent=args.max_concurrent,
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
    from rotomai.train.ppo import PPOConfig, run_ppo

    config = PPOConfig(
        init=args.init,
        out_dir=args.out_dir,
        battle_format=args.battle_format,
        # Populated at last: a teambuilt format (gen9ou, VGC) was previously unrunnable through
        # this subcommand -- `PPOConfig.team_pool` existed but nothing here ever set it, so every
        # real PPO run had to go through scripts/ppo_continue_gate.py.
        team_pool=args.team_pool,
        loop_penalty=args.loop_penalty,
        max_battle_turns=args.max_battle_turns,
        ent_coef_final=args.ent_coef_final,
        lr_final=args.lr_final,
        anneal_iters=args.anneal_iters,
        resume=args.resume,
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


def _print_live_summary(outcome: object) -> None:
    """Mirror of ``_print_result`` for a live session."""
    summary = outcome.summary  # type: ignore[attr-defined]
    finished, requested = summary["finished"], summary["requested"]
    print(f"\nlive session: {finished}/{requested} battles finished", end="")
    if summary["unfinished"]:
        print(f" ({summary['unfinished']} unfinished -- excluded from every metric)", end="")
    print()
    if not finished:
        print("  no finished battles: nothing to rate")
        return
    print(
        f"  W/L/T {summary['wins']}/{summary['losses']}/{summary['ties']}  "
        f"win_rate {summary['win_rate']:.3f}  score_rate {summary['score_rate']:.3f}  "
        f"mean_turns {summary['mean_turns']}"
    )
    print(f"  Glicko-1 {summary['glicko']} +/- {summary['glicko_rd']}   GXE {summary['gxe']:.4f}")
    print(f"  basis: {summary['gxe_basis']}")
    elo = summary["showdown_elo"]
    if elo["n_reported"]:
        print(f"  Showdown Elo (pre-battle): self {elo['self_first']} -> {elo['self_last']}, "
              f"opponent mean {elo['opponent_mean']} over {elo['n_reported']} rated games")
    if outcome.restarts:  # type: ignore[attr-defined]
        print(f"  reconnects: {outcome.restarts}")  # type: ignore[attr-defined]


async def _run_live(args: argparse.Namespace) -> None:
    from rotomai.live.session import LiveConfig, run_session

    cfg = LiveConfig(
        agent=args.agent, mode=args.mode, battle_format=args.battle_format, n=args.n,
        opponent=args.opponent, checkpoint=args.checkpoint, sample=args.sample,
        loop_penalty=args.loop_penalty, team_pool=args.team_pool, team_seed=args.team_seed,
        concurrency=args.concurrency, start_timer=args.start_timer,
        username=args.username, avatar=args.avatar,
        ws_url=args.server, auth_url=args.auth_url, allow_guest=args.allow_guest,
        save_replays=args.save_replays, out=args.out,
        battle_timeout=args.battle_timeout, stall_timeout=args.stall_timeout,
        login_timeout=args.login_timeout, max_restarts=args.max_restarts,
        ladder_ack=args.ladder_ack, log_level=args.log_level,
        use_live_ratings=args.use_live_ratings, opponent_rd=args.opponent_rd,
    )
    outcome = await run_session(cfg)
    _print_live_summary(outcome)
    print(f"\nwrote {args.out}")


async def _run_eval_ladder(args: argparse.Namespace) -> None:
    from pathlib import Path

    from rotomai.eval.ladder import (
        PairResult,
        cluster_bootstrap,
        format_table,
        parse_field,
        rate_field,
        run_round_robin,
        summarize_ladder,
        write_results,
    )

    entries = parse_field(args.field)
    labels = [entry.label for entry in entries]

    # PREFLIGHT, in the spirit of scripts/build26_analysis.sh: every reason to refuse is checked
    # BEFORE any battle is played. A bad anchor or a missing checkpoint discovered after 4,000
    # battles costs the whole run, and both are one string comparison away from being caught here.
    if args.anchor not in labels:
        raise SystemExit(
            f"--anchor {args.anchor!r} is not in the field ({', '.join(labels)}).\n"
            "The anchor fixes the additive gauge of the fit, so it must be an agent that plays."
        )
    missing = [
        entry.checkpoint
        for entry in entries
        if entry.checkpoint is not None and not Path(entry.checkpoint).exists()
    ]
    if missing:
        raise SystemExit("missing checkpoint(s):\n  " + "\n  ".join(missing))

    pairs_n = len(entries) * (len(entries) - 1) // 2
    print(
        f"[{args.battle_format}] eval ladder: {len(entries)} agents, {pairs_n} pairs "
        f"x {args.n} battles = {pairs_n * args.n} total"
    )
    for entry in entries:
        anchor = "  (anchor)" if entry.label == args.anchor else ""
        print(f"    {entry.label}{anchor}")
    print()

    def _progress(pair: PairResult) -> None:
        print(
            f"  {pair.a} vs {pair.b}: {pair.wins}-{pair.losses}-{pair.ties} "
            f"(score {pair.score:.3f})",
            flush=True,
        )

    tournament = await run_round_robin(
        entries,
        args.battle_format,
        args.n,
        team_pool=args.team_pool,
        concurrency=args.concurrency,
        sample=args.sample,
        loop_penalty=args.loop_penalty,
        on_pair=_progress,
    )
    pairs = tournament.pairs
    fit = rate_field(pairs, anchor=args.anchor)

    # The cluster bootstrap runs AFTER the battles and cannot fail the run: an interval is worth
    # having but not worth losing a completed tournament over, so a failure is reported and the
    # point estimate is still written.
    bootstrap = None
    if args.bootstrap > 0 and tournament.records:
        print(
            f"\n  cluster bootstrap: {args.bootstrap} resamples over team matchups...", flush=True
        )
        try:
            bootstrap = cluster_bootstrap(
                tournament.records, fit, anchor=args.anchor,
                n_resamples=args.bootstrap, seed=args.bootstrap_seed,
            )
        except Exception as exc:  # noqa: BLE001 -- never lose a finished tournament to the CI
            print(f"  bootstrap FAILED ({exc}); writing the point estimate only")

    summary = summarize_ladder(
        entries, pairs, fit, battle_format=args.battle_format, n_battles=args.n,
        bootstrap=bootstrap,
    )
    write_results(
        args.out,
        {
            "field_spec": args.field,
            "team_pool": args.team_pool,
            "concurrency": args.concurrency,
            "sample": args.sample,
            "loop_penalty": args.loop_penalty,
            "bootstrap_resamples": args.bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
        },
        summary,
    )
    print()
    print(format_table(summary))
    print(f"\n  basis: {summary['gxe_basis']}")
    if "warning" in summary:
        print(f"\n  WARNING: {summary['warning']}")
    if "unbounded_note" in summary:
        print(f"\n  NOTE: {summary['unbounded_note']}")
    print(f"\nwrote {args.out}")


def _run_fetch_replays(args: argparse.Namespace) -> None:
    from rotomai.data.replays import fetch_replays

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
    from rotomai.data.ingest import ingest_and_save
    from rotomai.data.replays import cached_replay_paths
    from rotomai.data.reward import RewardWeights

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
        complete_own_team=args.complete_own_team,
        impute_usage=args.impute_usage,
    )


def _run_resim_replays(args: argparse.Namespace) -> None:
    from rotomai.data.replays import cached_replay_paths
    from rotomai.data.resim import resim_and_save
    from rotomai.data.reward import RewardWeights

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
        "--format", dest="battle_format", default=DEFAULT_FORMAT, help="Showdown string format"
    )
    parser.add_argument(
        "--team-pool",
        default=None,
        help="packed team pool; REQUIRED on a teambuilt format and defaulted per format "
             "(gen9ou / VGC pools are committed under rotomai/teambuilding/data/)",
    )
    # Per SIDE, because a mirror is a legitimate matchup: `--p1 offrl --p2 offrl` with two
    # checkpoints is how one build is scored against another. Without these the only way to point a
    # learned agent at a downloaded weight was an agent-specific environment variable
    # (ROTOMAI_OFFRL_CHECKPOINT and friends), which cannot express two different ones at once.
    parser.add_argument("--p1-checkpoint", default=None, help="weights for --p1 if it is learned")
    parser.add_argument("--p2-checkpoint", default=None, help="weights for --p2 if it is learned")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rotomai")
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
    train.add_argument(
        "--pp-aug-frac",
        type=float,
        default=0.0,
        help="Build 10: fraction of attack-labeled deep-turn rows forced to full pp at train time",
    )
    train.add_argument(
        "--pp-aug-turn-threshold",
        type=float,
        default=0.15,
        help="Normalized turn above which a row counts as 'deep' for --pp-aug-frac",
    )
    train.add_argument(
        "--pp-noise-std",
        type=float,
        default=0.0,
        help="Build 11: Gaussian jitter std added to every present-move pp channel (train time)",
    )
    train.add_argument(
        "--pp-resample-frac",
        type=float,
        default=0.0,
        help="Build 11: fraction of present pp cells resampled from the batch pp pool (train time)",
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

    collect_rl.add_argument(
        "--team-pool",
        dest="team_pool",
        default=None,
        help="Packed-team pool for teambuilt formats (gen9ou, VGC); omit for Random Battles",
    )
    collect_rl.add_argument(
        "--max-battle-turns",
        dest="max_battle_turns",
        type=int,
        default=None,
        help="Forfeit a battle past this many recorded decisions (omit = no ceiling)",
    )

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
    # Arch flags default to None so an explicit request is distinguishable from silence:
    # an AC->AC warm-start adopts the checkpoint's arch, and a conflicting explicit
    # request must fail loud rather than be discarded (see offline_rl._arch_conflicts).
    train_rl.add_argument(
        "--d-model", type=int, default=None, help="Transformer token width (default: 128)"
    )
    train_rl.add_argument(
        "--n-layers", type=int, default=None, help="Transformer encoder layers (default: 2)"
    )
    train_rl.add_argument(
        "--n-heads", type=int, default=None, help="Transformer attention heads (default: 4)"
    )
    train_rl.add_argument(
        "--ff-dim", type=int, default=None, help="Transformer feed-forward width (default: 256)"
    )
    train_rl.add_argument(
        "--no-id-embed",
        dest="id_embed",
        action="store_false",
        help="entity_transformer: disable learned species/move/item/ability embeddings",
    )
    train_rl.set_defaults(id_embed=True)
    train_rl.add_argument(
        "--id-embed-init",
        default="random",
        choices=["random", "prior"],
        help="entity_transformer: 'prior' warm-starts species/move embeddings from dex features",
    )
    train_rl.add_argument("--n-bins", type=int, default=51)
    train_rl.add_argument("--beta", type=float, default=1.0, help="AWR temperature")
    train_rl.add_argument("--value-coef", type=float, default=0.5, help="Value-loss weight")
    train_rl.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    train_rl.add_argument("--seed", type=int, default=0, help="Init + train/val-split seed")

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
    selfplay.add_argument(
        "--max-concurrent", type=int, default=20, help="Concurrent battles in collection/eval"
    )
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
    selfplay.add_argument(
        "--team-pool",
        dest="team_pool",
        default=None,
        help="Packed-team pool for teambuilt formats (gen9ou, VGC); omit for Random Battles",
    )
    selfplay.add_argument(
        "--max-battle-turns",
        dest="max_battle_turns",
        type=int,
        default=None,
        help="Forfeit a battle past this many decisions (omit = no ceiling, the OU behavior)",
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

    # B6f: flags PPOConfig has carried for several builds that this subcommand never exposed, so
    # a teambuilt or scheduled run was reproducible only through scripts/ppo_continue_gate.py.
    ppo.add_argument(
        "--team-pool",
        dest="team_pool",
        default=None,
        help="Packed-team pool for teambuilt formats (gen9ou, VGC); omit for Random Battles",
    )
    ppo.add_argument(
        "--loop-penalty",
        dest="loop_penalty",
        type=float,
        default=0.0,
        help="Build-14 LoopGuard penalty on the learner/learned opponents (0 = off)",
    )
    ppo.add_argument(
        "--max-battle-turns",
        dest="max_battle_turns",
        type=int,
        default=None,
        help="Forfeit a battle past this many decisions (omit = no ceiling, the OU behavior)",
    )
    ppo.add_argument(
        "--ent-coef-final",
        type=float,
        default=None,
        help="Build-19: anneal ent_coef linearly to this by the final iter (omit = constant)",
    )
    ppo.add_argument(
        "--lr-final",
        type=float,
        default=None,
        help="Build-19: anneal lr linearly to this by the final iter (omit = constant)",
    )
    ppo.add_argument(
        "--anneal-iters",
        dest="anneal_iters",
        type=int,
        default=None,
        help="Anneal lr/ent_coef over this many iters instead of --iters; past it both hold at "
        "their finals (omit = anneal over --iters)",
    )
    ppo.add_argument(
        "--resume",
        action="store_true",
        help="Continue from --out-dir's last completed iteration instead of starting over; "
        "requires --anneal-iters so the schedule spans the same horizon in every chunk",
    )

    live = sub.add_parser(
        "live",
        help="M5 deploy: play on a LIVE Showdown server (challenge / accept / ladder)",
        description=(
            "Connect a trained agent to a live Showdown server under a DEDICATED BOT account. "
            f"Credentials come from ${USERNAME_ENV} / ${PASSWORD_ENV} and are never written to "
            "--out. Default mode is 'accept' (opt-in opponents only). 'ladder' plays the RANKED "
            "ladder, which plan.md NG3 puts out of scope -- see --ladder-ack."
        ),
    )
    live.add_argument("--agent", default="ppo", choices=sorted(AGENTS), help="Agent to deploy")
    live.add_argument(
        "--mode", default="accept", choices=["challenge", "accept", "ladder"],
        help="challenge: send --n challenges to --opponent; accept: accept --n incoming "
             "challenges; ladder: search --n RANKED ladder games (policy-gated)",
    )
    live.add_argument(
        "--opponent", default=None,
        help="challenge: username to challenge (REQUIRED). accept: comma-separated allowlist; "
             "omit to accept anyone",
    )
    live.add_argument("--n", type=int, default=1, help="Number of battles to play")
    live.add_argument(
        "--format", dest="battle_format", default=DEFAULT_FORMAT,
        help="Showdown format string. FIXED for the session: poke-env silently ignores "
             "challenges in any other format",
    )
    live.add_argument("--checkpoint", default=None, help="Checkpoint for a learned agent")
    live.add_argument("--sample", action="store_true", help="Sample instead of arg-max")
    live.add_argument("--loop-penalty", type=float, default=0.0, help="Build-14 LoopGuard")
    live.add_argument("--team-pool", default=None, help="Packed team pool for teambuilt formats")
    live.add_argument("--team-seed", type=int, default=0, help="TeamPool sampling seed")
    live.add_argument(
        "--concurrency", type=int, default=1,
        help="max_concurrent_battles. Keep at 1 live: the turn clock is per battle and every "
             "concurrent battle shares one process",
    )
    live.add_argument(
        "--username", default=None,
        help=f"Bot account name (default ${USERNAME_ENV}). The password is ALWAYS read from "
             f"${PASSWORD_ENV} -- never a flag, never logged, never written to --out",
    )
    live.add_argument("--avatar", default=None, help="Showdown avatar for the bot account")
    live.add_argument(
        "--server", default=None,
        help="Websocket URL override for a PRIVATE eval server (default: wss://sim3.psim.us)",
    )
    live.add_argument("--auth-url", default=None, help="Auth endpoint override (with --server)")
    live.add_argument(
        "--allow-guest", action="store_true",
        help="Permit an empty password. REFUSED against the public server: poke-env degrades to "
             "a Guest name and then hangs forever with no error",
    )
    live.add_argument(
        "--no-timer", dest="start_timer", action="store_false",
        help="Do NOT send '/timer on' at battle start",
    )
    live.set_defaults(start_timer=True)
    live.add_argument("--save-replays", default=None, help="Directory to save replays into")
    live.add_argument(
        "--out", default="results/live_session.json",
        help="Per-battle records + Glicko-1/GXE summary (rewritten after every battle)",
    )
    live.add_argument("--battle-timeout", type=float, default=900.0, help="Per-battle ceiling (s)")
    live.add_argument(
        "--stall-timeout", type=float, default=300.0,
        help="Seconds without forward progress before the connection is declared dead "
             "(poke-env 0.15 neither reconnects nor raises)",
    )
    live.add_argument("--login-timeout", type=float, default=30.0, help="Seconds to await login")
    live.add_argument("--max-restarts", type=int, default=3, help="Client rebuilds on a drop")
    live.add_argument(
        "--ladder-ack", default=None, choices=[LADDER_ACK],
        help="REQUIRED for --mode ladder. Acknowledges plan.md NG3 / section 15: automated RANKED "
             f"ladder play is out of scope. You must ALSO export {ALLOW_LADDER_ENV}=1. The "
             "acknowledgement is recorded in --out",
    )
    live.add_argument("--log-level", type=int, default=None, help="poke-env logger level")
    live.add_argument(
        "--use-live-ratings", action="store_true",
        help="Rate each opponent at its OBSERVED Showdown rating instead of pinning the field at "
             "1500/350. Only rated ladder games report one, so this is a no-op elsewhere. CAVEAT: "
             "Showdown's ladder line carries ELO and this treats it as a Glicko rating -- the "
             "units are not identical, and the results file records that it was used",
    )
    live.add_argument(
        "--opponent-rd", type=float, default=MAX_RD,
        help="Deviation assigned to an opponent's observed rating (default: the 350 ceiling, i.e. "
             "treat it as barely-known). Only meaningful with --use-live-ratings",
    )

    ladder = sub.add_parser(
        "eval-ladder",
        help="R-EVAL: rate a FIELD of agents against each other on the local server (G2's metric)",
        description=(
            "Play a round-robin over a field of agents and fit every rating jointly, so GXE and "
            "Glicko-1 are measured against a VARIED opponent field rather than one pinned "
            "baseline -- which is the only condition under which they carry information a win "
            "rate did not. This is the 'private/agent-only server or eval ladder' plan.md 12 asks "
            "for. It does NOT replace scripts/seed_strength_gate.py as the build-vs-build "
            "statistic, and its GXE is not comparable to a Showdown GXE (which is measured "
            "against humans)."
        ),
    )
    ladder.add_argument(
        "--field", default=",".join(DEFAULT_FIELD),
        help="Comma-separated agents as 'name' or 'name@checkpoint'. Default spans the measured "
             "strength range from random to the best PPO build",
    )
    ladder.add_argument("--n", type=int, default=150, help="Battles per pair")
    ladder.add_argument(
        "--anchor", default=DEFAULT_ANCHOR,
        help="Agent pinned at 1500 to fix the fit's additive gauge. A round-robin identifies "
             "rating DIFFERENCES only, so one agent must be pinned; the default is the project's "
             "fixed baseline, which makes the ratings commensurable with every published win rate",
    )
    ladder.add_argument(
        "--format", dest="battle_format", default=DEFAULT_FORMAT, help="Showdown format string"
    )
    ladder.add_argument(
        "--team-pool", default=None,
        help="Packed team pool, REQUIRED for teambuilt formats such as gen9ou",
    )
    ladder.add_argument(
        "--concurrency", type=int, default=20,
        help="max_concurrent_battles; the local server is the bottleneck",
    )
    ladder.add_argument("--sample", action="store_true", help="Sample instead of arg-max")
    ladder.add_argument(
        "--loop-penalty", type=float, default=DEFAULT_LOOP_PENALTY,
        help="Build-14 LoopGuard. Defaults to the value the AUTHORITATIVE strength gate uses (4), "
             "not 0, so the ladder scores the same policy the published win rates scored",
    )
    ladder.add_argument(
        "--bootstrap", type=int, default=300,
        help="Cluster-bootstrap resamples over TEAM MATCHUPS, for a 95%% CI on each rating. RD is "
             "binomial-only and therefore a LOWER BOUND on a teambuilt format, where battles "
             "cluster by matchup; this is the interval that is entitled to be read as one. "
             "0 disables it (the point estimate is unaffected either way)",
    )
    ladder.add_argument(
        "--bootstrap-seed", type=int, default=0, help="Resampling seed, so the CI is reproducible"
    )
    ladder.add_argument(
        "--out", default="results/eval_ladder.json",
        help="Standings, the full win matrix, and the basis they must be read with",
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
    ingest.add_argument(
        "--no-complete-own-team",
        dest="complete_own_team",
        action="store_false",
        help="Disable two-pass own-team completion (reproduce the v1 progressive-reveal POV)",
    )
    ingest.add_argument(
        "--no-impute-usage",
        dest="impute_usage",
        action="store_false",
        help="Disable usage-prior imputation of still-unrevealed own item/ability/moves",
    )
    ingest.set_defaults(complete_own_team=True, impute_usage=True)

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
    elif args.command == "live":
        asyncio.run(_run_live(args))
    elif args.command == "eval-ladder":
        asyncio.run(_run_eval_ladder(args))
    elif args.command == "fetch-replays":
        _run_fetch_replays(args)
    elif args.command == "ingest-replays":
        _run_ingest_replays(args)
    elif args.command == "resim-replays":
        _run_resim_replays(args)


if __name__ == "__main__":
    main()

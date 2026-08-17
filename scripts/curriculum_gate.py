"""Lever 13 gate: tougher-opponent AWR self-play curriculum from the GREEN checkpoint.

Across L10 (on-policy gradient), L11 (depth-1 search) and L12 (depth-2 search) nothing
compounds on the GREEN base -- a strong local-ceiling signal, but only against a *fixed*
opponent set. The one mechanism that ever cleared the wall was L9: AWR over all turns
(winners AND losers) at data scale -- changing the learning *signal*, not the inference
machinery. GREEN still *loses* to two fast rule-based bots stronger than itself
(simpleheuristics ~42.5%, heuristic ~41.7%). This lever re-runs the wall-clearing
mechanism (AWR, all turns, both sides recorded) on a tougher, self-generated opponent
distribution, powered up (the M4 self-play loop was underpowered + never run from GREEN).

``heuristic`` is held OUT of training (anchors = simpleheuristics + the self-play league);
we report win-rate vs the held-out heuristic, so a gain means generalization, not matchup
memorization -- directly comparable to the Lever 9 (47.7%) and Lever 10 gates.

    bash scripts/run_server.sh                       # local Showdown on :8000
    python scripts/curriculum_gate.py \
        --init checkpoints/offrl_scale_et_prior_s0.pt \
        --seeds 0,1,2 \
        --out results/curriculum_gate.json

Three gates, cheap-first:
  * Gate A -- a no-training KILL pre-flight: collect one small GREEN-vs-tough shard and
    confirm the AWR signal exists (winner/loser start-return gap + a non-trivial loser
    share) before paying for the full run. ``--preflight-only`` runs just this.
  * Gate B -- per seed, ``run_selfplay`` from GREEN; keep the full per-iter curve, pick
    the best iter by vs_heuristic, record final vs_iter0 (the unbiased compounding check
    L10 failed at 0.442).
  * Gate C -- a confirmatory ladder (n=300) on the single best checkpoint across all four
    baselines (only when Gate B looks promising).

Writes the grid to ``--out`` (JSON) and prints a summary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

import numpy as np

from rotomai.config import DEFAULT_FORMAT
from rotomai.data.collect import PlayerSpec, collect_selfplay
from rotomai.data.reward import RewardWeights
from rotomai.data.rl_dataset import discounted_returns
from rotomai.eval.arena import build_player, evaluate_built, policy_agent
from rotomai.train.selfplay import SelfPlayConfig, run_selfplay

_LADDER = ("random", "maxbasepower", "simpleheuristics", "heuristic")
_EVAL_CONCURRENCY = 20  # the local server is the bottleneck; keep many battles in flight

# Gate A thresholds: AWR needs a value signal that separates wins from losses, and the
# tough opponent must actually beat GREEN enough to leave winning lines in the data.
_GAP_MIN = 1.0  # winner-minus-loser mean start-return (L9 saw +5.48 on its shard)
_LOSER_FRAC_MIN = 0.15  # both classes must be well represented
_LOSER_FRAC_MAX = 0.85

# Gate B verdict bands vs the ~0.42-0.48 GREEN start.
_GREEN_VS_HEUR = 0.55  # best-iter mean vs_heuristic must clear this
_GREEN_VS_ITER0 = 0.55  # final policy must beat its own start head-to-head
_AMBER_VS_ITER0 = 0.52  # below this (with no clear vs_heuristic gain) is flat, not green
_RED_VS_HEUR = 0.40  # vs_heuristic well under this is a collapse


# --------------------------------------------------------------------------------------
# Gate A -- pre-flight signal check (pure stats, server-free + unit-testable).
# --------------------------------------------------------------------------------------
def preflight_stats(reward: np.ndarray, done: np.ndarray, gamma: float) -> dict:
    """Winner/loser start-return separation over a collected self-play shard.

    Episodes are the segments ending at ``done=True``; an episode is a *winner* when its
    terminal (last-turn) reward is positive. The per-turn discounted MC return at an
    episode's first turn is its full return, so the winner-minus-loser gap in those
    start-returns measures how cleanly the value target separates outcomes -- exactly the
    signal AWR re-weights on. A non-trivial loser fraction confirms the tough opponent
    leaves winning lines for AWR to imitate (both sides are recorded).
    """
    reward = np.asarray(reward, dtype=np.float64)
    done = np.asarray(done, dtype=bool)
    ret = discounted_returns(reward.astype(np.float32), done, gamma)
    ends = [int(i) for i in np.flatnonzero(done)]
    if not ends:
        return {
            "n_episodes": 0,
            "n_winners": 0,
            "n_losers": 0,
            "loser_fraction": 0.0,
            "mean_win_start": 0.0,
            "mean_loss_start": 0.0,
            "gap": 0.0,
        }
    starts = [0] + [e + 1 for e in ends[:-1]]
    win_starts = [float(ret[s]) for s, e in zip(starts, ends, strict=True) if reward[e] > 0]
    loss_starts = [float(ret[s]) for s, e in zip(starts, ends, strict=True) if reward[e] <= 0]
    mean_win = statistics.mean(win_starts) if win_starts else 0.0
    mean_loss = statistics.mean(loss_starts) if loss_starts else 0.0
    n_ep = len(ends)
    return {
        "n_episodes": n_ep,
        "n_winners": len(win_starts),
        "n_losers": len(loss_starts),
        "loser_fraction": round(len(loss_starts) / n_ep, 4),
        "mean_win_start": round(mean_win, 4),
        "mean_loss_start": round(mean_loss, 4),
        "gap": round(mean_win - mean_loss, 4),
    }


def preflight_verdict(stats: dict) -> str:
    """``PASS`` if the AWR signal is present and balanced, else ``KILL``."""
    if stats["n_winners"] == 0 or stats["n_losers"] == 0:
        return "KILL"
    if stats["gap"] < _GAP_MIN:
        return "KILL"
    if not (_LOSER_FRAC_MIN <= stats["loser_fraction"] <= _LOSER_FRAC_MAX):
        return "KILL"
    return "PASS"


async def run_preflight(
    init: str,
    fmt: str,
    games: int,
    max_concurrent: int,
    team_pool: str | None = None,
    max_battle_turns: int | None = None,
) -> dict:
    """Collect a small GREEN-vs-tough shard and report the AWR pre-flight signal."""
    # ASK, do not name. `build_player` refuses a singles-only name on a doubles format, and this
    # is the only self-play -> results/ bridge in the tree, so an M4 doubles arm has to come
    # through here. Same repair as `train.selfplay` and `rpredict_oppmodel_gate`.
    agent = policy_agent(fmt)
    learner = PlayerSpec(agent, checkpoint_path=init, sample=True)
    opponents = [
        PlayerSpec("simpleheuristics"),
        PlayerSpec(agent, checkpoint_path=init, sample=True),  # the iter-0 league member
    ]
    dataset = await collect_selfplay(
        learner,
        opponents,
        games,
        fmt,
        RewardWeights(),
        gamma=0.99,
        max_concurrent=max_concurrent,
        # Gate A collects through its OWN `collect_selfplay` call, not through `SelfPlayConfig`.
        # Threading the pool into the config alone left this one teamless, and the first VGC run
        # died here on a rejected-team popup after the job had already claimed its node.
        team_pool=team_pool,
        max_battle_turns=max_battle_turns,
    )
    stats = preflight_stats(dataset.reward, dataset.done, dataset.gamma)
    verdict = preflight_verdict(stats)
    stats["verdict"] = verdict
    stats["turns"] = int(len(dataset.reward))
    print(
        f"Gate A pre-flight: {stats['n_episodes']} episodes "
        f"({stats['n_winners']} win / {stats['n_losers']} loss, "
        f"loser_frac {stats['loser_fraction']:.2f}) | start-return "
        f"win {stats['mean_win_start']:.2f} - loss {stats['mean_loss_start']:.2f} "
        f"= gap {stats['gap']:.2f} | {verdict}"
    )
    return stats


# --------------------------------------------------------------------------------------
# Gate B -- the powered-up self-play run, per seed.
# --------------------------------------------------------------------------------------
def _out_dir(arm: str, seed: int) -> str:
    return f"checkpoints/{arm}_s{seed}"


def _selfplay_config(init: str, seed: int, args: argparse.Namespace) -> SelfPlayConfig:
    return SelfPlayConfig(
        init=init,
        out_dir=_out_dir(args.arm, seed),
        data_dir=f"data/{args.arm}_s{seed}",
        battle_format=args.battle_format,
        team_pool=args.team_pool,
        max_battle_turns=args.max_battle_turns,
        iters=args.iters,
        games_per_opp=args.games_per_opp,
        pop_size=args.pop_size,
        buffer_iters=args.buffer_iters,
        anchors=("simpleheuristics",),  # heuristic held OUT of training
        eval_baselines=("random", "simpleheuristics", "heuristic"),
        eval_n=args.eval_n,
        max_concurrent=args.max_concurrent,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=seed,
    )


def _best_checkpoint(out_dir: str, init: str, best_iter: int) -> str:
    """Path the best iteration's policy was written to (iter 0 == the warm-start ckpt)."""
    if best_iter == 0:
        return init
    return str(Path(out_dir) / f"iter_{best_iter:02d}.pt")


def _seed_record(init: str, seed: int, curve: list[dict], arm: str) -> dict:
    out_dir = _out_dir(arm, seed)
    start = next(p for p in curve if p["iter"] == 0)
    sp_points = [p for p in curve if p["iter"] >= 1]
    best = max(sp_points, key=lambda p: p.get("vs_heuristic", 0.0)) if sp_points else start
    final = curve[-1]
    start_wr = float(start.get("vs_heuristic", 0.0))
    best_wr = float(best.get("vs_heuristic", 0.0))
    return {
        "seed": seed,
        "out_dir": out_dir,
        "start_vs_heuristic": start_wr,
        "best_iter": int(best["iter"]),
        "best_vs_heuristic": best_wr,
        "best_checkpoint": _best_checkpoint(out_dir, init, int(best["iter"])),
        "final_vs_iter0": final.get("vs_iter0"),
        "delta_vs_start": round(best_wr - start_wr, 4),
        "curve": curve,
    }


def _stats(xs: list[float]) -> dict:
    return {
        "values": xs,
        "mean": statistics.mean(xs) if xs else 0.0,
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs) if xs else 0.0,
        "max": max(xs) if xs else 0.0,
    }


def _summarize(records: list[dict]) -> dict:
    return {
        "start_vs_heuristic": _stats([r["start_vs_heuristic"] for r in records]),
        "best_vs_heuristic": _stats([r["best_vs_heuristic"] for r in records]),
        "final_vs_iter0": _stats(
            [r["final_vs_iter0"] for r in records if r["final_vs_iter0"] is not None]
        ),
    }


def verdict(summary: dict) -> str:
    """GREEN / AMBER / RED over the seed summary (mirrors the L10 PPO-gate logic)."""
    best = summary["best_vs_heuristic"]["mean"]
    iter0 = summary["final_vs_iter0"]
    iter0_mean = iter0["mean"] if iter0["values"] else 0.0
    if best < _RED_VS_HEUR or (iter0["values"] and iter0_mean < 0.40):
        return "RED"
    if best >= _GREEN_VS_HEUR and iter0_mean > _GREEN_VS_ITER0:
        return "GREEN"
    return "AMBER"


def _print_table(summary: dict, verd: str) -> None:
    print(f"\n{'metric':>22}  {'mean+-std':>16}  {'min':>6}  {'max':>6}  (start ~0.42-0.48)")
    print("-" * 70)
    for key in ("start_vs_heuristic", "best_vs_heuristic", "final_vs_iter0"):
        s = summary[key]
        print(f"{key:>22}  {s['mean']:.3f}+-{s['std']:.3f}      {s['min']:.3f}  {s['max']:.3f}")
    print(f"\nVERDICT: {verd}")


# --------------------------------------------------------------------------------------
# Gate C -- confirmatory ladder on the single best checkpoint.
# --------------------------------------------------------------------------------------
async def _ladder(
    ckpt: str,
    fmt: str,
    n: int,
    team_pool: str | None = None,
    max_battle_turns: int | None = None,
) -> dict[str, float]:
    """Gate C. THE THIRD collection site in this file that needed the pool and did not have it.

    Gate A collects, the self-play loop collects, and this ladder builds its own players -- three
    independent paths, and threading the pool into `SelfPlayConfig` reached exactly one of them.
    `build_player` refuses a teamless teambuilt build now, so a fourth site cannot be added quietly.

    Both sides draw from the same pool with different seeds, as the ceiling gate does, so the
    matchup varies and the mirror stays fair in expectation.

    IT MUST ALSO RUN THE ARM'S CONFIGURATION, and it did not. `_eval_point` passes
    `max_battle_turns` to all four of its builds and this ladder passed none, so the first VGC
    capability record read the SAME checkpoint at 0.383 on the curve and 0.600 on the ladder --
    0.22 apart, against an n=60 standard error of ~0.063. A confirmatory ladder that confirms a
    different game than the one it is confirming is worse than no ladder.
    """
    def _team(seed: int) -> object | None:
        if not team_pool:
            return None
        from rotomai.teambuilding.pool import TeamPool

        return TeamPool.from_packed_file(team_pool, seed=seed)

    out: dict[str, float] = {}
    for base in _LADDER:
        learner = build_player(
            policy_agent(fmt), fmt, checkpoint_path=ckpt,
            max_concurrent_battles=_EVAL_CONCURRENCY,
            team=_team(0),  # type: ignore[arg-type]
            max_battle_turns=max_battle_turns,
        )
        opponent = build_player(
            base, fmt, max_concurrent_battles=_EVAL_CONCURRENCY,
            team=_team(1),  # type: ignore[arg-type]
            max_battle_turns=max_battle_turns,
        )
        out[base] = round(await evaluate_built(learner, opponent, n), 4)
    return out


async def run_gate(args: argparse.Namespace) -> dict:
    init = args.init
    if not Path(init).exists():
        raise SystemExit(f"GREEN warm-start checkpoint '{init}' not found.")
    print(
        f"init {init} | seeds {args.seeds} | iters {args.iters} | "
        f"games_per_opp {args.games_per_opp} | eval_n {args.eval_n}"
    )

    result: dict = {
        "init": init,
        "arm": args.arm,
        "opponent": "heuristic",
        # RECORDED so `merge_gate_seeds` and any later comparison can refuse a cross-format pool.
        # The RB record predates these keys and carries no `format` at all, which is exactly the
        # ambiguity that makes an unlabelled arm uncomparable.
        "format": args.battle_format,
        "team_pool": args.team_pool,
        "max_battle_turns": args.max_battle_turns,
    }

    # Gate A -- cheap KILL pre-flight.
    if not args.skip_preflight:
        pre = await run_preflight(
            init, args.battle_format, args.preflight_games, args.max_concurrent,
            team_pool=args.team_pool, max_battle_turns=args.max_battle_turns,
        )
        result["preflight"] = pre
        if pre["verdict"] == "KILL":
            print("\nGate A KILL: AWR signal absent -- not running the full self-play loop.")
            _write(args.out, result)
            return result
    if args.preflight_only:
        _write(args.out, result)
        return result

    # Gate B -- per-seed powered self-play.
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    records: list[dict] = []
    for seed in seeds:
        cfg = _selfplay_config(init, seed, args)
        curve = await run_selfplay(cfg)
        rec = _seed_record(init, seed, [dict(p) for p in curve], args.arm)
        records.append(rec)
        print(
            f"seed={seed} start={rec['start_vs_heuristic']:.3f} "
            f"best={rec['best_vs_heuristic']:.3f} (iter {rec['best_iter']}) "
            f"delta={rec['delta_vs_start']:+.3f} final_vs_iter0={rec['final_vs_iter0']}"
        )

    summary = _summarize(records)
    verd = verdict(summary)
    result.update(
        {
            "seeds": seeds,
            "iters": args.iters,
            "games_per_opp": args.games_per_opp,
            "eval_n": args.eval_n,
            "ladder_n": args.ladder_n,
            "records": records,
            "summary": summary,
            "verdict": verd,
        }
    )

    # Gate C -- confirmatory ladder, only when Gate B looks promising.
    overall_best = max(records, key=lambda r: r["best_vs_heuristic"])
    result["best_checkpoint"] = overall_best["best_checkpoint"]
    if verd != "AMBER" or summary["best_vs_heuristic"]["mean"] >= _AMBER_VS_ITER0:
        print(f"\nconfirmatory ladder (n={args.ladder_n}) on {overall_best['best_checkpoint']}")
        ladder = await _ladder(
            overall_best["best_checkpoint"], args.battle_format, args.ladder_n, args.team_pool,
            max_battle_turns=args.max_battle_turns,
        )
        print("  " + "  ".join(f"{k} {v:.3f}" for k, v in ladder.items()))
        result["confirmatory_ladder"] = ladder

    _write(args.out, result)
    _print_table(summary, verd)
    print(f"\nwrote {args.out}")
    return result


def _write(out: str, result: dict) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="curriculum_gate")
    parser.add_argument("--init", default="checkpoints/offrl_scale_et_prior_s0.pt")
    parser.add_argument("--out", default=None,
                        help="default results/curriculum_gate.json for the RB arm, else "
                             "results/curriculum_gate_<arm>.json")
    parser.add_argument("--arm", default="curriculum_et_prior",
                        help="arm name: the `arm` field, checkpoints/<arm>_s<seed>/ and data/")
    parser.add_argument("--team-pool", default=None,
                        help="packed team pool; REQUIRED on a teambuilt format (gen9ou, VGC)")
    parser.add_argument("--max-battle-turns", type=int, default=None,
                        help="B6f per-battle decision ceiling; None is exact identity")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    parser.add_argument("--iters", type=int, default=12, help="Self-play iterations per seed")
    parser.add_argument("--games-per-opp", type=int, default=48, help="Battles per opponent")
    parser.add_argument("--pop-size", type=int, default=3, help="League members sampled per iter")
    parser.add_argument("--buffer-iters", type=int, default=4, help="Sliding shard window")
    parser.add_argument("--epochs", type=int, default=6, help="Fine-tune epochs per iteration")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-n", type=int, default=200, help="Per-iter eval battles")
    parser.add_argument("--ladder-n", type=int, default=300, help="Confirmatory ladder battles")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Concurrent battles")
    parser.add_argument("--preflight-games", type=int, default=40, help="Gate A games per opp")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip Gate A")
    parser.add_argument("--preflight-only", action="store_true", help="Run Gate A only")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--format", dest="battle_format", default=DEFAULT_FORMAT)
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = (
            "results/curriculum_gate.json"
            if args.arm == "curriculum_et_prior"
            else f"results/curriculum_gate_{args.arm}.json"
        )
    # A teambuilt format with no pool starts no battles at all, and the failure arrives one full
    # collection later. `ppo_seed.slurm` refuses the same thing in its preflight.
    if "randombattle" not in args.battle_format and not args.team_pool:
        raise SystemExit(
            f"--team-pool is required on the teambuilt format {args.battle_format!r}; "
            f"try rotomai/teambuilding/data/teams_gen9vgc.packed"
        )

    asyncio.run(run_gate(args))


if __name__ == "__main__":
    main()

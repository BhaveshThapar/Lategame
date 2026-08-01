"""Lever 10 gate: on-policy PPO continuation from the GREEN offline checkpoint.

Lever 9 (value-RL at scale) was the first method to beat the heuristic band -- the
EntityTransformer + dex-prior AWR policy hit ~47.7% vs heuristic with a critic that finally
fits the value function (value-MAE 0.28). Every prior method was either off-policy AWR
(re-weights actions *already in the data* -> caps at the demonstrator ceiling) or PPO on a
*broken* MLP critic (M5 was flat/collapsed). For the first time we have both a >47% start and
a working low-variance GAE baseline, so on-policy PPO can push probability toward actions
outside the demonstrator distribution with a critic that can score them. This gate tests
whether that compounds past the AWR ceiling.

``heuristic`` is held OUT of training (anchors = simpleheuristics + the self-play league); we
report win-rate vs the held-out heuristic, so a gain means generalization, not matchup
memorization -- comparable to the Lever 9 gate.

    bash scripts/run_server.sh                       # local Showdown on :8000
    python scripts/ppo_continue_gate.py \
        --init checkpoints/offrl_scale_et_prior_s0.pt \
        --seeds 0,1,2 \
        --out results/ppo_continue_gate.json

Per seed it runs ``run_ppo`` from the GREEN checkpoint and keeps the full per-iter curve, then
runs a confirmatory ladder (n=300) on the single best PPO checkpoint across all four baselines.
Writes the grid to ``--out`` (JSON) and prints a summary. Gate vs the 47.7%+-2.5% start:
GREEN if best-iter mean vs_heuristic clears ~55% (band above the start) and final vs_iter0 >
0.50 with no collapse; AMBER if flat within noise; RED on collapse below the start.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from poke_env.teambuilder.teambuilder import Teambuilder

from lategame.config import DEFAULT_FORMAT
from lategame.eval.arena import build_player, evaluate_built
from lategame.teambuilding.pool import TeamPool
from lategame.train.ppo import PPOConfig, run_ppo

_LADDER = ("random", "maxbasepower", "simpleheuristics", "heuristic")
_EVAL_CONCURRENCY = 20  # the local server is the bottleneck; keep many battles in flight


def _ppo_config(
    init: str,
    seed: int,
    iters: int,
    games_per_opp: int,
    eval_n: int,
    device: str,
    fmt: str,
    team_pool: str | None,
    loop_penalty: float,
    ckpt_prefix: str,
    *,
    ent_coef: float = PPOConfig.ent_coef,
    ent_coef_final: float | None = None,
    lr: float = PPOConfig.lr,
    lr_final: float | None = None,
    target_kl: float = PPOConfig.target_kl,
    anneal_iters: int | None = None,
) -> PPOConfig:
    return PPOConfig(
        init=init,
        out_dir=f"checkpoints/{ckpt_prefix}_s{seed}",  # do not clobber the M5 ppo/ run
        battle_format=fmt,
        team_pool=team_pool,
        loop_penalty=loop_penalty,
        anchors=("simpleheuristics",),  # heuristic held OUT of training
        eval_baselines=("random", "simpleheuristics", "heuristic"),
        iters=iters,
        games_per_opp=games_per_opp,
        eval_n=eval_n,
        max_concurrent=_EVAL_CONCURRENCY,
        device=device,
        seed=seed,
        # Build 19: `*_final=None` holds the value constant == the Build 16-18 schedule.
        ent_coef=ent_coef,
        ent_coef_final=ent_coef_final,
        lr=lr,
        lr_final=lr_final,
        # Build 23: the KL early-stop bar (1.5 * this) was the ACTIVE constraint in v22 -- it bound
        # 59/150 iters, which is why that NULL was not attributable. Reachable from the CLI now.
        target_kl=target_kl,
        # Build 24: None == iters, so an unset run reproduces v17-v23 exactly.
        anneal_iters=anneal_iters,
    )


def _best_checkpoint(out_dir: str, init: str, best_iter: int) -> str:
    """Path the best iteration's policy was written to (iter 0 == the warm-start ckpt)."""
    if best_iter == 0:
        return init
    return str(Path(out_dir) / f"iter_{best_iter:02d}.pt")


def _seed_record(init: str, seed: int, curve: list[dict], ckpt_prefix: str) -> dict:
    out_dir = f"checkpoints/{ckpt_prefix}_s{seed}"
    start = next(p for p in curve if p["iter"] == 0)
    ppo_points = [p for p in curve if p["iter"] >= 1]
    best = max(ppo_points, key=lambda p: p.get("vs_heuristic", 0.0)) if ppo_points else start
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


async def _ladder(
    ckpt: str,
    fmt: str,
    n: int,
    team: str | Teambuilder | None = None,
    loop_penalty: float = 0.0,
) -> dict[str, float]:
    """Greedy win-rate of one checkpoint across all baselines (the confirmatory ladder)."""
    out: dict[str, float] = {}
    for base in _LADDER:
        learner = build_player(
            "offrl",
            fmt,
            checkpoint_path=ckpt,
            max_concurrent_battles=_EVAL_CONCURRENCY,
            team=team,
            loop_penalty=loop_penalty,
        )
        opponent = build_player(
            base, fmt, max_concurrent_battles=_EVAL_CONCURRENCY, team=team
        )
        out[base] = round(await evaluate_built(learner, opponent, n), 4)
    return out


def _summarize(records: list[dict]) -> dict:
    best = [r["best_vs_heuristic"] for r in records]
    iter0 = [r["final_vs_iter0"] for r in records if r["final_vs_iter0"] is not None]
    start = [r["start_vs_heuristic"] for r in records]

    def _stats(xs: list[float]) -> dict:
        return {
            "values": xs,
            "mean": statistics.mean(xs) if xs else 0.0,
            "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs) if xs else 0.0,
            "max": max(xs) if xs else 0.0,
        }

    return {
        "start_vs_heuristic": _stats(start),
        "best_vs_heuristic": _stats(best),
        "final_vs_iter0": _stats(iter0),
    }


def _print_table(summary: dict) -> None:
    print(f"\n{'metric':>22}  {'mean+-std':>16}  {'min':>6}  {'max':>6}  (start ~0.477+-0.025)")
    print("-" * 70)
    for key in ("start_vs_heuristic", "best_vs_heuristic", "final_vs_iter0"):
        s = summary[key]
        print(f"{key:>22}  {s['mean']:.3f}+-{s['std']:.3f}      {s['min']:.3f}  {s['max']:.3f}")


async def run_gate(
    init: str,
    out: str,
    seeds: list[int],
    iters: int,
    games_per_opp: int,
    eval_n: int,
    ladder_n: int,
    device: str,
    fmt: str,
    team_pool: str | None,
    loop_penalty: float,
    ckpt_prefix: str,
    *,
    ent_coef: float = PPOConfig.ent_coef,
    ent_coef_final: float | None = None,
    lr: float = PPOConfig.lr,
    lr_final: float | None = None,
    target_kl: float = PPOConfig.target_kl,
    anneal_iters: int | None = None,
) -> dict:
    if not Path(init).exists():
        raise SystemExit(f"GREEN warm-start checkpoint '{init}' not found.")
    print(
        f"init {init} | seeds {seeds} | iters {iters} | "
        f"games_per_opp {games_per_opp} | eval_n {eval_n} | "
        f"team_pool {team_pool} | loop_penalty {loop_penalty} | prefix {ckpt_prefix}"
    )
    # The gate JSON stores only the win-rate curve, so the trust region is reconstructed from the
    # log. Print the bar the run actually used -- ppo_telemetry.py's --kl-bar must match it.
    print(f"trust region: target_kl {target_kl} | kl_bar {1.5 * target_kl:.4g}")
    ent_end = ent_coef if ent_coef_final is None else ent_coef_final
    lr_end = lr if lr_final is None else lr_final
    # Build 24: name the anneal HORIZON, because it is not always `iters` any more and the whole
    # point of pinning it is a contrast that is unreadable if the log does not say which was used.
    horizon = anneal_iters or iters
    print(
        f"schedule: ent_coef {ent_coef} -> {ent_end} | lr {lr:.2e} -> {lr_end:.2e}"
        f" | anneal over {horizon} iters"
        f"{' (== iters)' if anneal_iters is None else ' (PINNED, iters=' + str(iters) + ')'}"
        f"{'  (constant == Build 18)' if ent_coef_final is None and lr_final is None else ''}"
    )
    # One shared pool: both sides of every eval/ladder battle draw from it (fair mirror).
    team = TeamPool.from_packed_file(team_pool) if team_pool else None

    records: list[dict] = []
    for seed in seeds:
        cfg = _ppo_config(
            init, seed, iters, games_per_opp, eval_n, device, fmt,
            team_pool, loop_penalty, ckpt_prefix,
            ent_coef=ent_coef, ent_coef_final=ent_coef_final, lr=lr, lr_final=lr_final,
            target_kl=target_kl, anneal_iters=anneal_iters,
        )
        curve = await run_ppo(cfg)
        rec = _seed_record(init, seed, curve, ckpt_prefix)
        records.append(rec)
        print(
            f"seed={seed} start={rec['start_vs_heuristic']:.3f} "
            f"best={rec['best_vs_heuristic']:.3f} (iter {rec['best_iter']}) "
            f"delta={rec['delta_vs_start']:+.3f} final_vs_iter0={rec['final_vs_iter0']}"
        )

    overall_best = max(records, key=lambda r: r["best_vs_heuristic"])
    print(f"\nconfirmatory ladder (n={ladder_n}) on {overall_best['best_checkpoint']}")
    ladder = await _ladder(overall_best["best_checkpoint"], fmt, ladder_n, team, loop_penalty)
    print("  " + "  ".join(f"{k} {v:.3f}" for k, v in ladder.items()))

    summary = _summarize(records)
    result = {
        "init": init,
        "arm": "ppo_et_prior",
        "format": fmt,
        "team_pool": team_pool,
        "loop_penalty": loop_penalty,
        "ckpt_prefix": ckpt_prefix,
        "ent_coef": ent_coef,
        "ent_coef_final": ent_coef_final,
        "lr": lr,
        "lr_final": lr_final,
        "target_kl": target_kl,
        "anneal_iters": anneal_iters,
        "seeds": seeds,
        "iters": iters,
        "games_per_opp": games_per_opp,
        "eval_n": eval_n,
        "ladder_n": ladder_n,
        "opponent": "heuristic",
        "best_checkpoint": overall_best["best_checkpoint"],
        "confirmatory_ladder": ladder,
        "records": records,
        "summary": summary,
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print_table(summary)
    print(f"\nwrote {out_path}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ppo_continue_gate")
    parser.add_argument("--init", default="checkpoints/offrl_scale_et_prior_s0.pt")
    parser.add_argument("--out", default="results/ppo_continue_gate.json")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    parser.add_argument("--iters", type=int, default=10, help="PPO iterations per seed")
    parser.add_argument("--games-per-opp", type=int, default=16, help="Rollout battles per opp")
    parser.add_argument("--eval-n", type=int, default=200, help="Per-iter eval battles")
    parser.add_argument("--ladder-n", type=int, default=300, help="Confirmatory ladder battles")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--format", dest="battle_format", default=DEFAULT_FORMAT)
    parser.add_argument(
        "--team-pool",
        dest="team_pool",
        default=None,
        help="Packed-team pool for teambuilt formats (gen9ou); omit for Random Battles",
    )
    parser.add_argument(
        "--loop-penalty",
        dest="loop_penalty",
        type=float,
        default=0.0,
        help="Build-14 LoopGuard penalty on the learner/learned opponents (0 = off)",
    )
    parser.add_argument(
        "--ckpt-prefix",
        dest="ckpt_prefix",
        default="ppo_scale_et_prior",
        help="checkpoints/<prefix>_s{seed}/ output dir (use ppo_ou_et_prior for OU)",
    )
    # Build 19: linear schedules over the run. Omitting the *-final flags holds the value
    # constant, reproducing Build 16-18 exactly.
    parser.add_argument("--ent-coef", type=float, default=PPOConfig.ent_coef)
    parser.add_argument(
        "--ent-coef-final",
        type=float,
        default=None,
        help="Build-19: anneal ent_coef linearly to this by the final iter (omit = constant)",
    )
    parser.add_argument("--lr", type=float, default=PPOConfig.lr)
    parser.add_argument(
        "--lr-final",
        type=float,
        default=None,
        help="Build-19: anneal lr linearly to this by the final iter (omit = constant)",
    )
    # Build 20-22 all ran at the PPOConfig default (0.03) because nothing forwarded it. v22's
    # trust region bound 59/150 iters and cost that build its attribution; omitting this flag
    # reproduces v20-v22 exactly.
    parser.add_argument(
        "--target-kl",
        dest="target_kl",
        type=float,
        default=PPOConfig.target_kl,
        help="Approx-KL early-stop target; the epoch loop stops past 1.5x this (omit = 0.03)",
    )
    # Build 24: --iters silently doubles as the lr/ent anneal horizon, so v22-vs-v23b is not
    # "the same optimizer run longer" -- at iteration 40 they sat at lr 9.08e-05 and 1.51e-04.
    # Pin this to compare budgets on update COUNT alone. Omitting it reproduces v17-v23.
    parser.add_argument(
        "--anneal-iters",
        dest="anneal_iters",
        type=int,
        default=None,
        help="Anneal lr/ent_coef over this many iters instead of --iters; past it both hold at "
        "their finals (omit = anneal over --iters, the Build 19-23 behavior)",
    )
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    asyncio.run(
        run_gate(
            args.init,
            args.out,
            seeds,
            args.iters,
            args.games_per_opp,
            args.eval_n,
            args.ladder_n,
            args.device,
            args.battle_format,
            args.team_pool,
            args.loop_penalty,
            args.ckpt_prefix,
            ent_coef=args.ent_coef,
            ent_coef_final=args.ent_coef_final,
            lr=args.lr,
            lr_final=args.lr_final,
            target_kl=args.target_kl,
            anneal_iters=args.anneal_iters,
        )
    )


if __name__ == "__main__":
    main()

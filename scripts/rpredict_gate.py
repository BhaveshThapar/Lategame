"""Lever 11 Gate B: does depth-1 search help on the frozen GREEN base?

Gate A proved the forward step is exact; the recon mini-gate proved determinization is
~100% observable-faithful. This is the payoff test: layer depth-1 expectimax (policy prior +
GREEN value-head leaves + a uniform/worst-case opponent over the determinized state) on the
*frozen* GREEN checkpoint and ask whether the lookahead beats the greedy base it descends from.

Same checkpoint powers both the ``search`` agent and the greedy ``offrl`` base, so any gap is
the search procedure, not the weights. We report, at the same n in one session:
  * search vs heuristic        vs.  base(offrl) vs heuristic   (the headline comparison)
  * search vs the greedy base  (head-to-head; >0.5 means lookahead adds value)
  * search vs random           (sanity)

    bash scripts/run_server.sh
    python scripts/rpredict_gate.py --init checkpoints/offrl_scale_et_prior_s0.pt --n 100

GREEN-ish (promising, promote to Gate C): search vs heuristic clears base vs heuristic with a
real margin AND search vs base > 0.5. AMBER: within noise. RED: below base.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from rotomai.config import DEFAULT_FORMAT
from rotomai.eval.arena import build_player, evaluate_built

_CONCURRENCY = 10  # search choose_move blocks on the node RPC, so keep this modest


async def _winrate(p1: str, p2: str, n: int, ckpt: str, fmt: str) -> float:
    a = build_player(p1, fmt, checkpoint_path=ckpt if p1 in ("search", "offrl") else None,
                     max_concurrent_battles=_CONCURRENCY)
    b = build_player(p2, fmt, checkpoint_path=ckpt if p2 in ("search", "offrl") else None,
                     max_concurrent_battles=_CONCURRENCY)
    return await evaluate_built(a, b, n)


def _set_search_env(args: argparse.Namespace) -> None:
    os.environ["ROTOMAI_SEARCH_CHECKPOINT"] = args.init
    os.environ["ROTOMAI_SEARCH_DETERMINIZATIONS"] = str(args.determinizations)
    os.environ["ROTOMAI_SEARCH_OPP_AGG"] = args.opp_agg
    os.environ["ROTOMAI_SEARCH_OPP_CAP"] = str(args.opp_cap)
    os.environ["ROTOMAI_SEARCH_BLEND"] = str(args.blend)
    os.environ["ROTOMAI_SEARCH_SHAPED"] = str(args.shaped)
    os.environ["ROTOMAI_SEARCH_DEPTH"] = str(args.depth)
    os.environ["ROTOMAI_SEARCH_TOPK_MY"] = str(args.top_k_my)
    os.environ["ROTOMAI_SEARCH_OPP_CAP_DEEP"] = str(args.opp_cap_deep)
    os.environ["ROTOMAI_SEARCH_SEED"] = str(args.seed)


async def run_gate(args: argparse.Namespace) -> dict:
    if not Path(args.init).exists():
        raise SystemExit(f"checkpoint '{args.init}' not found.")
    _set_search_env(args)
    cfg = {
        "determinizations": args.determinizations,
        "opp_agg": args.opp_agg,
        "opp_cap": args.opp_cap,
        "blend": args.blend,
        "shaped_coef": args.shaped,
        "depth": args.depth,
        "top_k_my": args.top_k_my,
        "opp_cap_deep": args.opp_cap_deep,
        "seed": args.seed,
    }
    print(f"Gate B: n={args.n}  search cfg={cfg}")

    fmt, ckpt = args.battle_format, args.init
    results = {}
    results["search_vs_random"] = await _winrate("search", "random", args.sanity_n, ckpt, fmt)
    print(f"  search vs random      = {results['search_vs_random']:.3f}  (sanity)")
    results["base_vs_heuristic"] = await _winrate("offrl", "heuristic", args.n, ckpt, fmt)
    print(f"  base  vs heuristic    = {results['base_vs_heuristic']:.3f}")
    results["search_vs_heuristic"] = await _winrate("search", "heuristic", args.n, ckpt, fmt)
    print(f"  search vs heuristic   = {results['search_vs_heuristic']:.3f}")
    results["search_vs_base"] = await _winrate("search", "offrl", args.n, ckpt, fmt)
    print(f"  search vs base (h2h)  = {results['search_vs_base']:.3f}")

    delta = results["search_vs_heuristic"] - results["base_vs_heuristic"]
    if delta > 0.03 and results["search_vs_base"] > 0.52:
        verdict = "PROMISING"
    elif delta < -0.03 or results["search_vs_base"] < 0.45:
        verdict = "RED"
    else:
        verdict = "AMBER"

    out = {"gate": "rpredict_gate_b", "init": args.init, "n": args.n, "config": cfg,
           "results": results, "delta_vs_base_at_heuristic": round(delta, 4), "verdict": verdict}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\ndelta (search-base) vs heuristic = {delta:+.3f}   VERDICT: {verdict}")
    print(f"wrote {out_path}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rpredict_gate")
    p.add_argument("--init", default="checkpoints/offrl_scale_et_prior_s0.pt")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--sanity-n", type=int, default=40)
    p.add_argument("--determinizations", type=int, default=1)
    p.add_argument("--opp-agg", default="mean", choices=["mean", "min"])
    p.add_argument("--opp-cap", type=int, default=6)
    p.add_argument("--blend", type=float, default=0.0)
    p.add_argument("--shaped", type=float, default=3.0, help="leaf = V + shaped*state_value")
    p.add_argument("--depth", type=int, default=1, help="plies of lookahead (2 = Lever 12)")
    p.add_argument("--top-k-my", type=int, default=3, help="prune our actions by prior, deep plies")
    p.add_argument("--opp-cap-deep", type=int, default=3, help="opp branching cap, deep plies")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/rpredict_gate_b.json")
    p.add_argument("--format", dest="battle_format", default=DEFAULT_FORMAT)
    args = p.parse_args(argv)
    asyncio.run(run_gate(args))


if __name__ == "__main__":
    main()

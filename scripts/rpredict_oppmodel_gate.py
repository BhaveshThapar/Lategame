"""Lever 14: does a *real opponent model* make search beat the base?

L11/L12 retired search with one named residual -- the opponent model was too weak (uniform /
worst-case over the determinized foe). The eval opponent is actually a fixed white-box heuristic,
so model it exactly and re-run the depth-2 gate with probability-weighted expectimax.

Two gates, cheap-first:

  Gate A -- opponent-model fidelity (no battles). Re-sim real replays, determinize each decision
    turn, and check (1) ``build_opp_pov`` reproduces the opponent's own observable team (the
    learned arm's prerequisite) and (2) ``WhiteBoxHeuristicOpponent`` always yields a decodable
    one-hot that agrees with the real ``HeuristicAgent`` on the reconstructed opponent POV. KILL
    if the decisive white-box arm can't be decoded; flag the learned arm if the POV is unfaithful.

        python scripts/rpredict_oppmodel_gate.py --gate a --limit 40

  Gate B -- the search re-run (server). Depth-2 expectimax with ``opp_aggregation="model"``, arms
    {whitebox, learned}, vs the same base/heuristic comparisons as ``rpredict_gate``. PROMISING
    iff search vs base > 0.52 AND (search - base) vs heuristic > 0.03 -- the L12 bar (0.500/-0.025).

        bash scripts/run_server.sh
        python scripts/rpredict_oppmodel_gate.py --gate b --arms whitebox,learned --n 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

_HP_TOL = 0.02


# --------------------------------------------------------------------------- #
# Gate A -- opponent-model fidelity (no server, node only).
# --------------------------------------------------------------------------- #


def _opp_pov_team_matches(opp_digest_p1: dict[str, Any], drv_p2: dict[str, Any]) -> list[bool]:
    """Per-mon observable agreement between the built opponent POV and the driver's p2 digest."""
    oks: list[bool] = []
    for sp, drv in drv_p2.items():
        pe = opp_digest_p1.get(sp)
        if pe is None:
            continue  # unrevealed filled slot -> not observable in the POV, skip
        oks.append(
            abs(float(pe["hp"]) - float(drv.get("hp", -9))) <= _HP_TOL
            and pe["status"] == drv.get("status", "")
            and bool(pe["fainted"]) == bool(drv.get("fainted"))
            and bool(pe["active"]) == bool(drv.get("active"))
        )
    return oks


def _heuristic_choice_on_pov(opp_battle: Any) -> str | None:
    """The eval HeuristicAgent's pick on the opponent's own POV, as a move/species id."""
    from poke_env import to_id_str
    from poke_env.battle import Move

    from lategame.agents.heuristic_agent import heuristic_pick

    pick = heuristic_pick(
        opp_battle.active_pokemon,
        opp_battle.opponent_active_pokemon,
        opp_battle.available_moves,
        opp_battle.available_switches,
    )
    if pick is None:
        return None
    obj = pick[1]
    return to_id_str(obj.id if isinstance(obj, Move) else obj.species)


def run_gate_a(args: argparse.Namespace) -> dict[str, Any]:
    from poke_env import to_id_str

    from lategame.data.replays import iter_cached_replays
    from lategame.data.resim import _parse_inputlog_meta, run_driver
    from lategame.search.determinize import battle_to_spec
    from lategame.search.forward import ForwardModel
    from lategame.search.opponent_model import WhiteBoxHeuristicOpponent, build_opp_pov
    from lategame.search.recon_check import _decision_snapshots

    replays = list(iter_cached_replays(args.cache_dir))
    if not replays:
        raise SystemExit(f"no cached replays under '{args.cache_dir}'.")
    if 0 < args.limit < len(replays):
        replays = random.Random(args.seed).sample(replays, args.limit)

    prepared: list[tuple[str, str, int, str, str]] = []
    for rep in replays:
        il = rep.get("inputlog")
        if isinstance(il, str) and il:
            rid = str(rep.get("id") or f"r{len(prepared)}")
            p1n, p2n, gen = _parse_inputlog_meta(il)
            prepared.append((rid, il, gen, p1n, p2n))
    results = {
        str(r.get("id")): r
        for r in run_driver(({"id": rid, "inputlog": il} for rid, il, _, _, _ in prepared))
    }

    wb = WhiteBoxHeuristicOpponent()
    pov_oks: list[bool] = []
    decodable = agree = agree_total = snapshots = errored = 0
    fm = ForwardModel()
    try:
        for rid, _il, gen, p1n, _p2n in prepared:
            res = results.get(rid)
            if not res or "error" in res:
                continue
            raw = res.get("p1")
            events: list[Any] = raw if isinstance(raw, list) else []
            seen = 0
            for snap in _decision_snapshots(events, p1n, f"{rid}-p1", gen):
                if args.max_turns and seen >= args.max_turns:
                    break
                seen += 1
                snapshots += 1
                try:
                    recon = fm.reconstruct(battle_to_spec(snap, seed=snapshots))
                    opp_battle = build_opp_pov(recon, tag=f"{rid}-opp", gen=gen)
                except Exception:  # noqa: BLE001
                    errored += 1
                    continue
                opp_choices = recon.get("p2_choices") or []
                if opp_battle is not None:
                    # opp POV role p2 -> its own team is digest["p1"]; compare to the driver's
                    # ground-truth p2 (the determinized opponent's own observable state).
                    pov_oks.extend(
                        _opp_pov_team_matches(_digest(opp_battle)["p1"], recon["digest"]["p2"])
                    )
                # white-box: always a decodable one-hot over the offered choices?
                dist = wb.distribution(snap, opp_choices)
                one_hot = [c for c, p in dist.items() if p >= 0.999]
                if len(dist) == 1 and one_hot and one_hot[0] in {c["choice"] for c in opp_choices}:
                    decodable += 1
                    # agreement with the real heuristic on the reconstructed opponent POV.
                    if opp_battle is not None and opp_battle.active_pokemon is not None:
                        hc = _heuristic_choice_on_pov(opp_battle)
                        chosen = next(c for c in opp_choices if c["choice"] == one_hot[0])
                        if hc is not None:
                            agree_total += 1
                            agree += int(to_id_str(str(chosen["id"])) == hc)
    finally:
        fm.close()

    n_wb = max(1, snapshots - errored)
    pov_rate = (sum(pov_oks) / len(pov_oks)) if pov_oks else 0.0
    out = {
        "gate": "rpredict_oppmodel_gate_a",
        "snapshots": snapshots,
        "errored": errored,
        "opp_pov_team_fidelity": round(pov_rate, 4),
        "whitebox_decodable_rate": round(decodable / n_wb, 4),
        "whitebox_vs_heuristic_agreement": round(agree / agree_total, 4) if agree_total else None,
        "agreement_n": agree_total,
        "learned_arm_ok": pov_rate >= args.pov_threshold,
        "gate_a_pass": (decodable / n_wb) >= args.decode_threshold,
    }
    _write(out, args.out)
    print(json.dumps(out, indent=2))
    print(
        f"\nGate A: white-box decodable {out['whitebox_decodable_rate']:.3f} "
        f"(pass>={args.decode_threshold}) -> {'PASS' if out['gate_a_pass'] else 'KILL'};  "
        f"opp-POV fidelity {pov_rate:.3f} (learned arm "
        f"{'OK' if out['learned_arm_ok'] else 'DEGRADED -> uniform fallback'})"
    )
    return out


def _digest(battle: Any) -> dict[str, Any]:
    from lategame.search.determinize import pokeenv_digest

    return pokeenv_digest(battle)


# --------------------------------------------------------------------------- #
# Gate B -- the search re-run (server).
# --------------------------------------------------------------------------- #

# Search choose_move is SYNCHRONOUS and blocks poke-env's shared event loop for its full
# duration; with too many concurrent battles the loop is starved past the websocket keepalive
# window (~20s) and the server drops every connection at once. Keep concurrency modest --
# especially the learned arm, whose per-node opponent-POV deepcopies are heavier.
_CONCURRENCY = 6


def _set_search_env(args: argparse.Namespace, opp_model: str) -> None:
    os.environ["LATEGAME_SEARCH_CHECKPOINT"] = args.init
    os.environ["LATEGAME_SEARCH_OPP_MODEL"] = opp_model
    os.environ["LATEGAME_SEARCH_OPP_AGG"] = "model"
    os.environ["LATEGAME_SEARCH_DEPTH"] = str(args.depth)
    os.environ["LATEGAME_SEARCH_DETERMINIZATIONS"] = str(args.determinizations)
    os.environ["LATEGAME_SEARCH_OPP_CAP"] = str(args.opp_cap)
    os.environ["LATEGAME_SEARCH_OPP_CAP_DEEP"] = str(args.opp_cap_deep)
    os.environ["LATEGAME_SEARCH_TOPK_MY"] = str(args.top_k_my)
    os.environ["LATEGAME_SEARCH_SHAPED"] = str(args.shaped)
    os.environ["LATEGAME_SEARCH_SEED"] = str(args.seed)


async def _winrate(p1: str, p2: str, n: int, ckpt: str, fmt: str, concurrency: int) -> float:
    from lategame.eval.arena import build_player, evaluate_built

    a = build_player(p1, fmt, checkpoint_path=ckpt if p1 in ("search", "offrl") else None,
                     max_concurrent_battles=concurrency)
    b = build_player(p2, fmt, checkpoint_path=ckpt if p2 in ("search", "offrl") else None,
                     max_concurrent_battles=concurrency)
    return await evaluate_built(a, b, n)


async def run_gate_b(args: argparse.Namespace) -> dict[str, Any]:
    from lategame.config import DEFAULT_FORMAT

    if not Path(args.init).exists():
        raise SystemExit(f"checkpoint '{args.init}' not found.")
    fmt = args.battle_format or DEFAULT_FORMAT
    ckpt = args.init

    cc = args.concurrency
    base_vs_heur = await _winrate("offrl", "heuristic", args.n, ckpt, fmt, cc)
    print(f"base vs heuristic = {base_vs_heur:.3f}  (shared reference, concurrency={cc})")

    arms: dict[str, Any] = {}
    for opp_model in [a.strip() for a in args.arms.split(",") if a.strip()]:
        _set_search_env(args, opp_model)
        print(f"\n=== arm: opp_model={opp_model} (depth={args.depth}, concurrency={cc}) ===")
        r: dict[str, Any] = {
            "search_vs_random": await _winrate("search", "random", args.sanity_n, ckpt, fmt, cc),
            "search_vs_heuristic": await _winrate("search", "heuristic", args.n, ckpt, fmt, cc),
            "search_vs_base": await _winrate("search", "offrl", args.n, ckpt, fmt, cc),
        }
        delta = r["search_vs_heuristic"] - base_vs_heur
        if delta > 0.03 and r["search_vs_base"] > 0.52:
            verdict = "PROMISING"
        elif delta < -0.03 or r["search_vs_base"] < 0.45:
            verdict = "RED"
        else:
            verdict = "AMBER"
        r["delta_vs_base_at_heuristic"] = round(delta, 4)
        r["verdict"] = verdict
        arms[opp_model] = r
        print(f"  search vs random   = {r['search_vs_random']:.3f}  (sanity)")
        print(f"  search vs heuristic= {r['search_vs_heuristic']:.3f}  (base {base_vs_heur:.3f})")
        print(f"  search vs base h2h = {r['search_vs_base']:.3f}")
        print(f"  delta vs heuristic = {delta:+.3f}   VERDICT: {verdict}")

    out = {
        "gate": "rpredict_oppmodel_gate_b",
        "init": args.init,
        "n": args.n,
        "depth": args.depth,
        "base_vs_heuristic": round(base_vs_heur, 4),
        "arms": arms,
    }
    _write(out, args.out)
    print(f"\nwrote {args.out}")
    return out


def _write(out: dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rpredict_oppmodel_gate")
    p.add_argument("--gate", choices=["a", "b"], required=True)
    p.add_argument("--init", default="checkpoints/offrl_scale_et_prior_s0.pt")
    # Gate A
    p.add_argument("--cache-dir", default="replays/gen9randombattle")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--max-turns", type=int, default=6, help="snapshots per replay (0=all)")
    p.add_argument("--pov-threshold", type=float, default=0.85)
    p.add_argument("--decode-threshold", type=float, default=0.95)
    # Gate B
    p.add_argument("--arms", default="whitebox,learned")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--sanity-n", type=int, default=40)
    p.add_argument("--concurrency", type=int, default=_CONCURRENCY,
                   help="battles per event loop; lower avoids keepalive drops on slow search")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--determinizations", type=int, default=1)
    p.add_argument("--opp-cap", type=int, default=6)
    p.add_argument("--opp-cap-deep", type=int, default=3)
    p.add_argument("--top-k-my", type=int, default=3)
    p.add_argument("--shaped", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--format", dest="battle_format", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    if args.out is None:
        args.out = f"results/rpredict_oppmodel_gate_{args.gate}.json"
    if args.gate == "a":
        run_gate_a(args)
    else:
        asyncio.run(run_gate_b(args))


if __name__ == "__main__":
    main()

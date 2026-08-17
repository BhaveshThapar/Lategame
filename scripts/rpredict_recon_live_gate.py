"""Gate A' -- reconstruction fidelity on a format that has no inputlog, measured from LIVE play.

`scripts/rpredict_recon_gate.py` scores determinization by re-simulating replays. That path is
gen9-RB-only in practice: gen9ou public replays carry **no `inputlog`** (OU Build 2, which is why
OU ingest had to use the seed-free reconstructor), and no VGC replays are scraped at all. So the
teambuilt and doubles formats are scored from live battles instead -- every decision turn of a real
local-server game is a POV to determinize and compare.

The criteria are `recon_check._compare`, unchanged, so the numbers are directly comparable to the
published RB read (145,400 checks, 7 mismatches, 0.99995 match rate). What is checked is what the
agent OBSERVES: our full team, the revealed opponent mons, field and hazards. Hidden opponent
details are guesses by construction and are not scored.

**THIS GATES THE STRENGTH RUN.** Search on an unfaithful forward model is GIGO, which is why
Gate A runs first and why a low match rate is a finding rather than a nuisance.

    bash scripts/run_server.sh 8500
    ROTOMAI_SHOWDOWN_PORT=8500 python scripts/rpredict_recon_live_gate.py \
        --format gen9ou --n 20 --out results/rpredict_recon_ou.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from poke_env.battle import AbstractBattle
from poke_env.player import BattleOrder

from rotomai.config import is_doubles_format
from rotomai.search.recon_check import ReconStats, check_live_snapshot

#: Match rate at or above which the reconstruction is trusted enough to search on. The RB gate
#: measured 0.99995; this is deliberately looser because a teambuilt opponent's set is SAMPLED,
#: so a field that depends on the sampled set can legitimately differ.
PASS_RATE = 0.99


def _is_full_decision(battle: AbstractBattle) -> bool:
    """Whether every slot has an active Pokemon -- i.e. this is a MOVE turn, not a replacement.

    On doubles a fainted mon leaves its slot empty until it is replaced, and poke-env asks for that
    replacement through the same ``choose_move`` callback. A reconstruction cannot represent "slot
    2 is momentarily empty": the spec names the actives and the driver leads them out, so a
    one-active POV reconstructs as a two-active battle with an arbitrary bench mon promoted --
    which Gate A'' correctly reported as ~1 spurious `active` mismatch per snapshot.

    Search never runs at these turns either: a forced replacement has no move to look ahead over.
    So they are not sampled, rather than being sampled and scored as failures.
    """
    active = getattr(battle, "active_pokemon", None)
    if isinstance(active, list):
        return len(active) > 0 and all(m is not None for m in active)
    return active is not None


def _snapshotting_agent(base_cls: type, stats: ReconStats, model: Any, cap: int) -> type:
    """Wrap an agent so every decision it makes is also a reconstruction-fidelity sample."""

    class _Snapshotter(base_cls):
        def choose_move(self, battle: AbstractBattle) -> BattleOrder:
            if stats.snapshots < cap and _is_full_decision(battle):
                check_live_snapshot(stats, battle, model)
            return super().choose_move(battle)

    return _Snapshotter


async def _run(args: argparse.Namespace, stats: ReconStats, model: Any) -> None:
    from rotomai.eval.arena import AGENTS, build_player
    from rotomai.teambuilding.pool import TeamPool

    team = None
    if args.team_pool:
        team = TeamPool.from_packed_file(args.team_pool)

    watched = _snapshotting_agent(AGENTS[args.agent], stats, model, args.max_snapshots)
    original = AGENTS[args.agent]
    AGENTS[args.agent] = watched
    try:
        p1 = build_player(args.agent, args.battle_format, team=team, max_concurrent_battles=1)
    finally:
        AGENTS[args.agent] = original
    p2 = build_player(
        args.opponent,
        args.battle_format,
        team=TeamPool.from_packed_file(args.team_pool, seed=1) if args.team_pool else None,
        max_concurrent_battles=1,
    )
    # Concurrency 1 on purpose: one ForwardModel subprocess serves every snapshot, so parallel
    # battles would only queue on its pipe while making the progress line meaningless.
    await p1.battle_against(p2, n_battles=args.n)
    for player in (p1, p2):
        await player.ps_client.stop_listening()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", dest="battle_format", default="gen9ou")
    p.add_argument("--n", type=int, default=20, help="battles to play")
    p.add_argument("--agent", default="heuristic", help="the POV whose decisions are sampled")
    p.add_argument("--opponent", default="heuristic")
    p.add_argument("--team-pool", default=None, help="REQUIRED for a teambuilt format")
    p.add_argument("--max-snapshots", type=int, default=1500)
    p.add_argument("--showdown-dir", default="third_party/pokemon-showdown")
    p.add_argument("--node", default="node")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    if args.team_pool is None and "random" not in args.battle_format:
        raise SystemExit(f"{args.battle_format} is teambuilt -- pass --team-pool")
    out = args.out or f"results/rpredict_recon_{args.battle_format}.json"

    from rotomai.search.forward import ForwardModel

    stats = ReconStats()
    model = ForwardModel(showdown_dir=args.showdown_dir, node=args.node)
    try:
        asyncio.run(_run(args, stats, model))
    finally:
        model.close()

    payload: dict[str, Any] = {
        "gate": "rpredict_recon_live",
        "format": args.battle_format,
        "doubles": is_doubles_format(args.battle_format),
        "battles": args.n,
        "agent": args.agent,
        "opponent": args.opponent,
        "team_pool": args.team_pool,
        "pass_rate": PASS_RATE,
        **stats.to_dict(),
    }
    rate = float(payload.get("match_rate") or 0.0)
    payload["verdict"] = "PASS" if rate >= PASS_RATE and stats.snapshots else "FAIL"
    payload["note"] = (
        "Live-play reconstruction fidelity, scored on recon_check._compare -- the same criteria "
        "as the replay-driven RB gate (145,400 checks / 7 mismatches / 0.99995), so the numbers "
        "are comparable. Gates the search strength run: search on an unfaithful forward model is "
        "GIGO. Hidden opponent set details are SAMPLED from the usage prior and are not scored; "
        "only what the agent observes is."
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "by_field"}, indent=2))
    print(f"\nwrote {out}")
    if payload["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

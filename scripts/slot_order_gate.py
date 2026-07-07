"""OU Pivot Build 5 -- Gate A: does move-slot ORDER actually diverge train-vs-live?

Builds 2-4 densified the own-team POV and the agent got monotonically WORSE vs random
(v1 0.495 -> v3 0.13-0.45 -> v4 0.05-0.27). The Build-4 diagnosis: the action space picks
moves POSITIONALLY (slots 6-9 = ``list(active.moves.values())[:4]``, the same order the
encoder writes move blocks), but the ``moves`` dict insertion order differs train vs live:

  * train (log reconstruction): reveal order, then ``_complete_own_team`` appends the
    backfilled/imputed remainder in ``sorted(move_id)`` order;
  * live (``|request|``): the request's declaration order, all four moves from turn 1.

So slot k can mean a different move at train vs eval, and every densification step scrambles
more slots -- a label-semantics mismatch that would explain the monotone degradation. Before
any code change, this gate measures whether the scramble exists at magnitude:

  * train side: run the exact v4 training path (two-pass completion + usage-prior
    imputation) over cached gen9ou replays with a probe on ``ingest._move_sample``; at each
    resolvable move decision compare the used move's slot under the insertion order vs the
    canonical order (sorted by move id) that Build 5 would adopt.
  * live side: the OU eval teampool's packed teams carry the request declaration order
    directly (mon field 4 = comma-separated move ids); measure how often it differs from
    canonical.

KILL (do NOT build the canonical codec): ``move_label_scramble_rate`` < 0.10 AND
``live_mon_scramble_rate`` < 0.50 -- the orderings already agree, slot order cannot be the
failure, the lever is dead cheaply. Secondary check: canonical ``[:4]`` truncation must not
create material new drops on >4-move sets (``out_of_canonical4 - out_of_insertion4`` <=
0.005), else the truncation rule needs a rethink before the codec change. Sanity: >= 1000
move decisions.

    python scripts/slot_order_gate.py --cache-dir replays/gen9ou --sample 200

Gate B2's functionality read (needs the local server + trained v5 checkpoints) reuses this
script so the whole build's evidence lives in one place: per-seed offrl-vs-random win rates,
the exact signature that exposed the v1-v4 stall (losing to random).

    python scripts/slot_order_gate.py --stage b2-random \\
        --checkpoints checkpoints/offrl_gen9ou_v5_s0.pt,... --n 100 \\
        --out results/gateb_v5_vs_random.json
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from poke_env import to_id_str

from lategame.data import ingest
from lategame.data.ingest import _gen_from_log, _impute_kits, _prescan_kits, _reconstruct_pov
from lategame.data.replays import cached_replay_paths
from lategame.data.reward import RewardWeights
from lategame.data.usage_prior import load_usage_prior
from lategame.teambuilding.pool import DEFAULT_OU_POOL

_RESULTS = Path("results/slot_order_gate.json")

# KILL thresholds (pre-registered).
MIN_LABEL_SCRAMBLE = 0.10  # train-side: used move's slot moves under canonicalization
MIN_LIVE_SCRAMBLE = 0.50  # live-side: teampool declaration order differs from canonical
MAX_NEW_DROP_DELTA = 0.005  # canonical [:4] must not create material new drops (>4 moves)
MIN_DECISIONS = 1000


def canonical_move_ids(ids: list[str]) -> list[str]:
    """The Build-5 canonical slot order for a known-move id list: sorted, first four."""
    return sorted(ids)[:4]


def packed_move_lists(text: str) -> list[list[str]]:
    """Per-mon move-id lists (declaration order) from a packed-team file's text.

    Packed format: one team per line, mons joined by ``]``, fields by ``|``; field 4 is the
    comma-separated move list in the order the teambuilder declares them -- exactly the order
    a live ``|request|`` presents.
    """
    mons: list[list[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for mon in line.split("]"):
            fields = mon.split("|")
            if len(fields) > 4 and fields[4]:
                mons.append([to_id_str(m) for m in fields[4].split(",") if m])
    return mons


def mean_displacement(ids: list[str]) -> float:
    """Mean |declaration slot - canonical slot| over one mon's moves."""
    if not ids:
        return 0.0
    canonical = sorted(ids)
    return float(np.mean([abs(i - canonical.index(m)) for i, m in enumerate(ids)]))


class _SlotProbe:
    """Per-decision slot comparison between insertion order and canonical order."""

    def __init__(self) -> None:
        self.decisions = 0  # resolvable move decisions (used move known to the mon)
        self.kept = 0  # decisions the current pipeline labels successfully
        self.order_scrambled = 0  # insertion[:4] != canonical[:4] as sequences
        self.in_both = 0  # used move present in both 4-lists
        self.label_scrambled = 0  # ...and its slot index differs
        self.out_of_insertion4 = 0  # used move beyond insertion[:4] (current drops)
        self.out_of_canonical4 = 0  # used move beyond canonical[:4] (would-be drops)

    def classify(self, battle: Any, parts: list[str], kept: bool) -> None:
        active = battle.active_pokemon
        if active is None or len(parts) < 4:
            return
        used = to_id_str(parts[3])
        if used not in active.moves:
            return
        ids = [m.id for m in active.moves.values()]
        insertion = ids[:4]
        canonical = canonical_move_ids(ids)
        self.decisions += 1
        self.kept += kept
        if insertion != canonical:
            self.order_scrambled += 1
        in_ins, in_can = used in insertion, used in canonical
        self.out_of_insertion4 += not in_ins
        self.out_of_canonical4 += not in_can
        if in_ins and in_can:
            self.in_both += 1
            if insertion.index(used) != canonical.index(used):
                self.label_scrambled += 1

    def rates(self) -> dict[str, float]:
        n, b = self.decisions, self.in_both
        return {
            "n_move_decisions": float(n),
            "n_kept": float(self.kept),
            "move_label_scramble_rate": self.label_scrambled / b if b else 0.0,
            "mon_order_scramble_rate": self.order_scrambled / n if n else 0.0,
            "out_of_insertion4_rate": self.out_of_insertion4 / n if n else 0.0,
            "out_of_canonical4_rate": self.out_of_canonical4 / n if n else 0.0,
        }


@contextmanager
def _probe_slots() -> Iterator[_SlotProbe]:
    """Wrap ``ingest._move_sample`` to compare slot orders at each move decision.

    Classifies after the original sampler runs, so two-pass completion (which the sampler
    invokes) has populated the moveset -- the exact state labels and obs are built from.
    Classifies dropped decisions too (the out-of-4 metrics ARE the drop comparison); the
    ``kits is not None`` guard skips pass-1 prescan reconstructions.
    """
    probe = _SlotProbe()
    orig_move = ingest._move_sample

    def move(battle: Any, parts: list[str], tera: bool, weights: Any, kits: Any) -> Any:
        out = orig_move(battle, parts, tera, weights, kits)
        if kits is not None:
            probe.classify(battle, parts, kept=out is not None)
        return out

    ingest._move_sample = move
    try:
        yield probe
    finally:
        ingest._move_sample = orig_move


def _measure_train(replays: list[dict[str, Any]], weights: RewardWeights) -> dict[str, float]:
    """Reconstruct every POV down the v4 training path (completion + imputation), probed."""
    prior = load_usage_prior("gen9ou")
    if prior is None:  # the gate must measure the path training actually uses
        raise SystemExit("no usage-prior artifact for gen9ou -- run scripts/build_usage_prior.py")
    povs = fatal = 0
    with _probe_slots() as probe:
        for replay in replays:
            log = replay.get("log")
            players = replay.get("players")
            if not isinstance(log, str) or not isinstance(players, list) or len(players) != 2:
                continue
            lines = log.split("\n")
            gen = _gen_from_log(log)
            tag = str(replay.get("id") or "r")
            for username in players:
                povs += 1
                pov_tag = f"{tag}-{username}"
                try:
                    kits = _prescan_kits(lines, str(username), pov_tag, gen, weights)
                    _impute_kits(kits, pov_tag, prior)
                    _reconstruct_pov(lines, str(username), pov_tag, gen, weights, kits=kits)
                except Exception:  # noqa: BLE001 -- one bad POV must not abort the gate
                    fatal += 1
                    continue
    metrics = probe.rates()
    metrics["povs"] = float(povs)
    metrics["parse_rate"] = (povs - fatal) / povs if povs else 0.0
    return metrics


def _measure_live_pool(pool_path: str | Path) -> dict[str, float]:
    """Declaration-vs-canonical divergence over the eval teampool's mons."""
    mons = packed_move_lists(Path(pool_path).read_text(encoding="utf-8"))
    scrambled = sum(1 for m in mons if m != sorted(m))
    return {
        "n_teams": float(len(Path(pool_path).read_text(encoding="utf-8").strip().splitlines())),
        "n_mons": float(len(mons)),
        "mon_scramble_rate": scrambled / len(mons) if mons else 0.0,
        "mean_displacement": float(np.mean([mean_displacement(m) for m in mons])) if mons else 0.0,
    }


async def _run_b2_random(args: argparse.Namespace) -> None:
    """Per-seed offrl-vs-random win rates on the eval teampool (v4-record JSON shape)."""
    from lategame.eval.arena import build_player, evaluate_built
    from lategame.teambuilding.pool import TeamPool

    teams = TeamPool.from_packed_file(args.team_pool).teams
    seeds = itertools.count()

    def _pool() -> TeamPool:  # fresh pool per player so p1/p2 draw independently
        return TeamPool(teams, seed=next(seeds))

    checkpoints = [c for c in args.checkpoints.split(",") if c]
    rates: dict[str, float] = {}
    for idx, ckpt in enumerate(checkpoints):
        p1 = build_player(
            "offrl", args.format, checkpoint_path=ckpt,
            max_concurrent_battles=args.concurrency, team=_pool(),
        )
        p2 = build_player(
            "random", args.format, max_concurrent_battles=args.concurrency, team=_pool()
        )
        rate = await evaluate_built(p1, p2, args.n)
        rates[f"s{idx}"] = float(rate)
        print(f"  B2 vs-random s{idx} ({ckpt}): {rate:.3f}  (n={args.n})")

    result = {"vs_random": rates, "n": args.n, "format": args.format, "checkpoints": checkpoints}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("a", "b2-random"), default="a")
    ap.add_argument("--cache-dir", default="replays/gen9ou", help="Cached gen9ou replay JSON dir")
    ap.add_argument("--sample", type=int, default=200, help="Max replays to score")
    ap.add_argument("--team-pool", default=str(DEFAULT_OU_POOL), help="Packed eval teampool")
    ap.add_argument("--checkpoints", default="", help="b2-random: comma-separated offrl ckpts")
    ap.add_argument("--n", type=int, default=100, help="b2-random: battles per checkpoint")
    ap.add_argument("--format", default="gen9ou", help="b2-random: battle format")
    ap.add_argument("--concurrency", type=int, default=20, help="b2-random: concurrent battles")
    ap.add_argument("--out", default=str(_RESULTS))
    args = ap.parse_args()

    if args.stage == "b2-random":
        if not args.checkpoints:
            raise SystemExit("--stage b2-random requires --checkpoints")
        if args.out == str(_RESULTS):
            args.out = "results/gateb_v5_vs_random.json"
        asyncio.run(_run_b2_random(args))
        return

    paths = cached_replay_paths(args.cache_dir)[: args.sample]
    replays: list[dict[str, Any]] = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                replays.append(json.load(fh))
        except (OSError, ValueError):
            continue
    if not replays:
        raise SystemExit(f"no cached replays under {args.cache_dir} -- scrape first")

    train = _measure_train(replays, RewardWeights())
    live = _measure_live_pool(args.team_pool)

    new_drop_delta = train["out_of_canonical4_rate"] - train["out_of_insertion4_rate"]
    checks = {
        "scramble_exists": (
            train["move_label_scramble_rate"] >= MIN_LABEL_SCRAMBLE
            or live["mon_scramble_rate"] >= MIN_LIVE_SCRAMBLE
        ),
        "canonical_drop_delta_ok": new_drop_delta <= MAX_NEW_DROP_DELTA,
        "sample_ok": train["n_move_decisions"] >= MIN_DECISIONS,
    }
    verdict = "PASS" if all(checks.values()) else "KILL"

    result = {
        "n_replays": len(replays),
        "train": train,
        "live_pool": live,
        "new_drop_delta": new_drop_delta,
        "thresholds": {
            "min_label_scramble": MIN_LABEL_SCRAMBLE,
            "min_live_scramble": MIN_LIVE_SCRAMBLE,
            "max_new_drop_delta": MAX_NEW_DROP_DELTA,
            "min_decisions": MIN_DECISIONS,
        },
        "checks": checks,
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== Build 5 slot-order Gate A ({len(replays)} replays) ===")
    print(
        f"train  label scramble {train['move_label_scramble_rate']:.3f} "
        f"(kill-side < {MIN_LABEL_SCRAMBLE}); mon order scramble "
        f"{train['mon_order_scramble_rate']:.3f}"
    )
    print(
        f"       out-of-4: insertion {train['out_of_insertion4_rate']:.4f} vs canonical "
        f"{train['out_of_canonical4_rate']:.4f}  delta {new_drop_delta:+.4f} "
        f"(<= {MAX_NEW_DROP_DELTA})"
    )
    print(
        f"       {int(train['n_move_decisions'])} move decisions "
        f"({int(train['n_kept'])} kept) over {int(train['povs'])} POVs, "
        f"parse rate {train['parse_rate']:.3f}"
    )
    print(
        f"live   pool mon scramble {live['mon_scramble_rate']:.3f} "
        f"(kill-side < {MIN_LIVE_SCRAMBLE}); mean displacement "
        f"{live['mean_displacement']:.2f} slots over {int(live['n_mons'])} mons"
    )
    print(f"\nVERDICT: {verdict}  -> {args.out}")
    if verdict == "KILL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

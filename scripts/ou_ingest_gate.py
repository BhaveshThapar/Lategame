"""OU Pivot Build 2 -- Gate A: is log-based OU reconstruction faithful enough to train on?

Gen9-RB replays carry a PRNG seed, so ``data.resim`` recovers the full private POV by
re-simulating. Gen9-OU public replays carry **no ``inputlog``** (confirmed: keys are
format/id/log/players/rating/...), so there is no seed and no packed team -- resim is
impossible. But OU logs open with ``|poke|`` team-preview lines naming all six species
per side, so the *v1 log-based* ``data.ingest`` path is the right reconstructor once it
seeds those species from turn 0 (``ingest._register_preview``). This gate measures
whether that reconstruction is faithful before we spend anything on training.

This upgrade (Build 3) checks the channels the encoder actually feeds the model -- the
identity-embedding IDs (item/ability/move) -- not just species presence. Build 2's Gate A
passed on species but was blind to the own-team *detail* gap that made the trained agent OOD:
the log reveals the player's own item/ability/moves only progressively, while the live
``|request|`` hands over the full team from turn 1. ``ingest`` now closes this with two-pass
own-team completion (``_prescan_kits`` + ``_complete_own_team``); this gate measures how far
that goes on the exact channels that matter.

Measurements over a sample of cached gen9ou replays (no server, no training):

  * parse rate      -- POV-episodes reconstructed without a fatal error.
  * species coverage -- mean over decisions of (ego + opp species present)/12, from the
                        per-Pokemon "present" flag. Preview should push this to ~1.0.
  * drop rate       -- decisions seen but undecodable (forced/first-reveal-nickname switches).
  * reward sign     -- winner-POV mean episode return must exceed loser-POV mean.
  * species control (teeth) -- re-run with ``|poke|`` stripped; coverage must drop materially.
  * own-active detail channels -- over decisions, the own active mon's item-known rate,
                        ability-known rate, and move-count (/4), read from the encoder's ID
                        channels via ``OBS_LAYOUT``. Reported WITH two-pass and WITH it OFF
                        (the v1 progressive-reveal control); the ON-vs-OFF *lift* is the teeth,
                        and the absolute ON value is the log *ceiling* (residual = ideal - ON,
                        where ideal = live 1.0/1.0/4.0).

Known residual: ability is essentially irreducible from public logs -- poke-env already
auto-assigns single-option abilities, so the unknowns are multi-ability species whose ability
never triggered. If Gate B (functional test) underperforms, the low item/ability ceiling is the
trigger to escalate to usage-prior (Smogon) imputation -- the documented next lever.

KILL (do NOT train): parse rate < 0.80, species coverage < 0.95, drop rate > 0.40, reward gap
  <= 0, species control lift < 0.10, OR two-pass fails to lift the channels logs *can* fix
  (item-known lift < 0.05 or move-count lift < 0.30 vs the no-completion control).

    python scripts/ou_ingest_gate.py --cache-dir replays/gen9ou --sample 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lategame.data.ingest import (
    _episode_rewards,
    _gen_from_log,
    _prescan_kits,
    _reconstruct_pov,
)
from lategame.data.replays import cached_replay_paths, fetch_replays
from lategame.data.reward import RewardWeights
from lategame.features.encoder import OBS_LAYOUT

_RESULTS = Path("results/ou_ingest_gate.json")
_PDIM = OBS_LAYOUT.pokemon_dim  # each of the first 12 obs blocks is one Pokemon
# Trailing per-Pokemon ID channels (POKEMON_ID_FIELDS = species, item, ability).
_ITEM_CH = _PDIM - 2
_ABILITY_CH = _PDIM - 1

# KILL thresholds.
MIN_PARSE_RATE = 0.80
MIN_COVERAGE = 0.95
MAX_DROP_RATE = 0.40
MIN_COVERAGE_LIFT = 0.10  # preview must lift coverage vs the stripped-|poke| control
MIN_ITEM_LIFT = 0.05  # two-pass must lift own-active item-known vs the no-completion control
MIN_MOVE_LIFT = 0.30  # two-pass must lift own-active move-count vs the no-completion control
IDEAL_ITEM = 1.0  # live request reveals every own item...
IDEAL_ABILITY = 1.0  # ...ability...
IDEAL_MOVES = 4.0  # ...and full moveset from turn 1 (residual = ideal - ceiling)


def _present_counts(obs: np.ndarray) -> tuple[int, int]:
    """(ego, opp) species present in one decision obs, from each block's present flag."""
    ego = sum(1 for i in range(6) if obs[i * _PDIM] > 0.5)
    opp = sum(1 for i in range(6, 12) if obs[i * _PDIM] > 0.5)
    return ego, opp


def _active_detail(obs: np.ndarray) -> tuple[float, float, float] | None:
    """Own active mon's (item-known, ability-known, move-count/4) from the encoder ID channels.

    Finds the own active block by its active flag (channel 1), then reads its trailing item/
    ability ID channels (nonzero = a real vocab id = known) and counts the active-move blocks
    whose move-ID channel is nonzero. Returns ``None`` if no own mon is active this decision.
    """
    for i in range(6):
        if obs[i * _PDIM + 1] > 0.5:  # active flag
            item = 1.0 if obs[i * _PDIM + _ITEM_CH] > 0.5 else 0.0
            ability = 1.0 if obs[i * _PDIM + _ABILITY_CH] > 0.5 else 0.0
            ms, md = OBS_LAYOUT.moves_start, OBS_LAYOUT.move_dim
            moves = sum(1.0 for j in range(4) if obs[ms + j * md + md - 1] > 0.5)
            return item, ability, moves
    return None


def _strip_preview(lines: list[str]) -> list[str]:
    """Drop ``|poke|`` team-preview lines -> RB-style progressive reveal (negative control)."""
    return [ln for ln in lines if not ln.startswith("|poke|")]


def _score(
    replays: list[dict[str, Any]],
    weights: RewardWeights,
    strip: bool = False,
    complete: bool = True,
) -> dict[str, float]:
    """Reconstruct every POV; return coverage, detail channels + health counters.

    ``strip`` drops ``|poke|`` preview (species negative control); ``complete`` toggles two-pass
    own-team completion (``False`` = the v1 progressive-reveal detail control).
    """
    povs = fatal = 0
    turns = dropped = 0
    cov_sum = cov_n = 0.0
    item_sum = ability_sum = moves_sum = detail_n = 0.0
    winner_returns: list[float] = []
    loser_returns: list[float] = []

    for replay in replays:
        log = replay.get("log")
        players = replay.get("players")
        if not isinstance(log, str) or not isinstance(players, list) or len(players) != 2:
            continue
        lines = log.split("\n")
        if strip:
            lines = _strip_preview(lines)
        gen = _gen_from_log(log)
        tag = str(replay.get("id") or "r")
        for username in players:
            povs += 1
            pov_tag = f"{tag}-{username}"
            try:
                kits = (
                    _prescan_kits(lines, str(username), pov_tag, gen, weights)
                    if complete
                    else None
                )
                battle, records, values, drop = _reconstruct_pov(
                    lines, str(username), pov_tag, gen, weights, kits=kits
                )
            except Exception:  # noqa: BLE001 -- one bad POV must not abort the gate
                fatal += 1
                continue
            dropped += drop
            turns += len(records)
            for obs, _, _ in records:
                ego, opp = _present_counts(obs)
                cov_sum += (ego + opp) / 12.0
                cov_n += 1.0
                detail = _active_detail(obs)
                if detail is not None:
                    item_sum += detail[0]
                    ability_sum += detail[1]
                    moves_sum += detail[2]
                    detail_n += 1.0
            rewards = _episode_rewards(battle, records, values, weights)
            if rewards:
                ret = float(sum(rewards))
                (winner_returns if battle.won else loser_returns).append(ret)

    seen = turns + dropped
    return {
        "povs": float(povs),
        "parse_rate": (povs - fatal) / povs if povs else 0.0,
        "turns": float(turns),
        "drop_rate": dropped / seen if seen else 0.0,
        "species_coverage": cov_sum / cov_n if cov_n else 0.0,
        "item_known": item_sum / detail_n if detail_n else 0.0,
        "ability_known": ability_sum / detail_n if detail_n else 0.0,
        "move_count": moves_sum / detail_n if detail_n else 0.0,
        "winner_return": float(np.mean(winner_returns)) if winner_returns else 0.0,
        "loser_return": float(np.mean(loser_returns)) if loser_returns else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="replays/gen9ou", help="Cached gen9ou replay JSON dir")
    ap.add_argument("--sample", type=int, default=200, help="Max replays to score")
    ap.add_argument(
        "--fetch",
        type=int,
        default=0,
        help="If >0 and cache is thin, fetch this many gen9ou replays first",
    )
    ap.add_argument("--min-rating", type=int, default=1200)
    ap.add_argument("--out", default=str(_RESULTS))
    args = ap.parse_args()

    if args.fetch > 0:
        fetch_replays(
            battle_format="gen9ou",
            min_rating=args.min_rating,
            limit=args.fetch,
            cache_dir=args.cache_dir,
            max_pages=200,
        )

    paths = cached_replay_paths(args.cache_dir)[: args.sample]
    replays: list[dict[str, Any]] = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                replays.append(json.load(fh))
        except (OSError, ValueError):
            continue
    if not replays:
        raise SystemExit(f"no cached replays under {args.cache_dir} -- scrape first (or --fetch N)")

    weights = RewardWeights()
    main_m = _score(replays, weights, strip=False, complete=True)  # real training reconstruction
    species_ctrl = _score(replays, weights, strip=True, complete=True)  # species lift control
    detail_ctrl = _score(replays, weights, strip=False, complete=False)  # v1 no-completion control

    coverage = main_m["species_coverage"]
    lift = coverage - species_ctrl["species_coverage"]
    reward_gap = main_m["winner_return"] - main_m["loser_return"]
    item_lift = main_m["item_known"] - detail_ctrl["item_known"]
    ability_lift = main_m["ability_known"] - detail_ctrl["ability_known"]
    move_lift = main_m["move_count"] - detail_ctrl["move_count"]

    checks = {
        "parse_rate_ok": main_m["parse_rate"] >= MIN_PARSE_RATE,
        "coverage_ok": coverage >= MIN_COVERAGE,
        "drop_rate_ok": main_m["drop_rate"] <= MAX_DROP_RATE,
        "reward_sign_ok": reward_gap > 0.0,
        "species_control_has_teeth": lift >= MIN_COVERAGE_LIFT,
        "item_completion_has_teeth": item_lift >= MIN_ITEM_LIFT,
        "move_completion_has_teeth": move_lift >= MIN_MOVE_LIFT,
    }
    verdict = "PASS" if all(checks.values()) else "KILL"

    # Absolute ON-value = log ceiling; residual = live-ideal - ceiling. The ability residual is
    # inherent to public logs -> escalation trigger to usage-prior imputation if Gate B is weak.
    residual = {
        "item": IDEAL_ITEM - main_m["item_known"],
        "ability": IDEAL_ABILITY - main_m["ability_known"],
        "moves": IDEAL_MOVES - main_m["move_count"],
    }

    result = {
        "n_replays": len(replays),
        "with_completion": main_m,
        "stripped_preview_control": species_ctrl,
        "no_completion_control": detail_ctrl,
        "coverage_lift": lift,
        "reward_gap": reward_gap,
        "detail_lift": {"item": item_lift, "ability": ability_lift, "moves": move_lift},
        "residual_vs_live": residual,
        "thresholds": {
            "min_parse_rate": MIN_PARSE_RATE,
            "min_coverage": MIN_COVERAGE,
            "max_drop_rate": MAX_DROP_RATE,
            "min_coverage_lift": MIN_COVERAGE_LIFT,
            "min_item_lift": MIN_ITEM_LIFT,
            "min_move_lift": MIN_MOVE_LIFT,
        },
        "checks": checks,
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== OU ingest Gate A ({len(replays)} replays) ===")
    print(f"parse rate     {main_m['parse_rate']:.3f}   (>= {MIN_PARSE_RATE})")
    print(
        f"species cover  {coverage:.3f}   (>= {MIN_COVERAGE}); "
        f"stripped {species_ctrl['species_coverage']:.3f}; "
        f"lift {lift:+.3f} (>= {MIN_COVERAGE_LIFT})"
    )
    print(f"drop rate      {main_m['drop_rate']:.3f}   (<= {MAX_DROP_RATE})")
    print(
        f"reward sign    winner {main_m['winner_return']:+.3f} vs loser "
        f"{main_m['loser_return']:+.3f}  gap {reward_gap:+.3f} (> 0)"
    )
    print("own-active detail channels (ON two-pass | OFF v1 | lift | residual-vs-live):")
    print(
        f"  item-known   {main_m['item_known']:.3f} | {detail_ctrl['item_known']:.3f} | "
        f"{item_lift:+.3f} (>= {MIN_ITEM_LIFT}) | {residual['item']:.3f}"
    )
    print(
        f"  ability-known {main_m['ability_known']:.3f} | {detail_ctrl['ability_known']:.3f} | "
        f"{ability_lift:+.3f} (reported) | {residual['ability']:.3f}"
    )
    print(
        f"  move-count   {main_m['move_count']:.3f} | {detail_ctrl['move_count']:.3f} | "
        f"{move_lift:+.3f} (>= {MIN_MOVE_LIFT}) | {residual['moves']:.3f}"
    )
    print(f"turns kept     {int(main_m['turns'])} over {int(main_m['povs'])} POV-episodes")
    print(f"\nVERDICT: {verdict}  -> {args.out}")
    if verdict == "KILL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

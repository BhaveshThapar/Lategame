"""OU Pivot Build 2 -- Gate A: is log-based OU reconstruction faithful enough to train on?

Gen9-RB replays carry a PRNG seed, so ``data.resim`` recovers the full private POV by
re-simulating. Gen9-OU public replays carry **no ``inputlog``** (confirmed: keys are
format/id/log/players/rating/...), so there is no seed and no packed team -- resim is
impossible. But OU logs open with ``|poke|`` team-preview lines naming all six species
per side, so the *v1 log-based* ``data.ingest`` path is the right reconstructor once it
seeds those species from turn 0 (``ingest._register_preview``). This gate measures
whether that reconstruction is faithful before we spend anything on training.

Measurements over a sample of cached gen9ou replays (no server, no training):

  * parse rate      -- POV-episodes reconstructed without a fatal error.
  * species coverage -- mean over decisions of (ego + opp species present)/12, read from
                        the encoder's per-Pokemon "present" flag. Preview should push
                        this to ~1.0 from the first turn.
  * drop rate       -- decisions seen but undecodable (forced/first-reveal-nickname switches).
  * reward sign     -- winner-POV mean episode return must exceed loser-POV mean (the
                        shaped terminal reward must carry the right sign).
  * negative control (teeth) -- re-run with ``|poke|`` lines stripped (RB-style progressive
                        reveal). Coverage must drop materially; a metric that can't fall is
                        not a gate (Lever 11 lesson).

KILL (do NOT train; the residual is fidelity -> next lever is two-pass own-team completion):
  parse rate < 0.80, or species coverage < 0.95, or drop rate > 0.40, or reward gap <= 0,
  or the negative control fails to lower coverage (lift < 0.10).

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
    _reconstruct_pov,
)
from lategame.data.replays import cached_replay_paths, fetch_replays
from lategame.data.reward import RewardWeights
from lategame.features.encoder import OBS_LAYOUT

_RESULTS = Path("results/ou_ingest_gate.json")
_PDIM = OBS_LAYOUT.pokemon_dim  # each of the first 12 obs blocks is one Pokemon

# KILL thresholds.
MIN_PARSE_RATE = 0.80
MIN_COVERAGE = 0.95
MAX_DROP_RATE = 0.40
MIN_COVERAGE_LIFT = 0.10  # preview must lift coverage vs the stripped-|poke| control


def _present_counts(obs: np.ndarray) -> tuple[int, int]:
    """(ego, opp) species present in one decision obs, from each block's present flag."""
    ego = sum(1 for i in range(6) if obs[i * _PDIM] > 0.5)
    opp = sum(1 for i in range(6, 12) if obs[i * _PDIM] > 0.5)
    return ego, opp


def _strip_preview(lines: list[str]) -> list[str]:
    """Drop ``|poke|`` team-preview lines -> RB-style progressive reveal (negative control)."""
    return [ln for ln in lines if not ln.startswith("|poke|")]


def _score(replays: list[dict[str, Any]], weights: RewardWeights, strip: bool) -> dict[str, float]:
    """Reconstruct every POV; return coverage + health counters (optionally preview-stripped)."""
    povs = fatal = 0
    turns = dropped = 0
    cov_sum = cov_n = 0.0
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
            try:
                battle, records, values, drop = _reconstruct_pov(
                    lines, str(username), f"{tag}-{username}", gen, weights
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
    main_m = _score(replays, weights, strip=False)
    neg_m = _score(replays, weights, strip=True)

    coverage = main_m["species_coverage"]
    lift = coverage - neg_m["species_coverage"]
    reward_gap = main_m["winner_return"] - main_m["loser_return"]

    checks = {
        "parse_rate_ok": main_m["parse_rate"] >= MIN_PARSE_RATE,
        "coverage_ok": coverage >= MIN_COVERAGE,
        "drop_rate_ok": main_m["drop_rate"] <= MAX_DROP_RATE,
        "reward_sign_ok": reward_gap > 0.0,
        "negative_control_has_teeth": lift >= MIN_COVERAGE_LIFT,
    }
    verdict = "PASS" if all(checks.values()) else "KILL"

    result = {
        "n_replays": len(replays),
        "with_preview": main_m,
        "stripped_preview_control": neg_m,
        "coverage_lift": lift,
        "reward_gap": reward_gap,
        "thresholds": {
            "min_parse_rate": MIN_PARSE_RATE,
            "min_coverage": MIN_COVERAGE,
            "max_drop_rate": MAX_DROP_RATE,
            "min_coverage_lift": MIN_COVERAGE_LIFT,
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
        f"stripped {neg_m['species_coverage']:.3f}; lift {lift:+.3f} (>= {MIN_COVERAGE_LIFT})"
    )
    print(f"drop rate      {main_m['drop_rate']:.3f}   (<= {MAX_DROP_RATE})")
    print(
        f"reward sign    winner {main_m['winner_return']:+.3f} vs loser "
        f"{main_m['loser_return']:+.3f}  gap {reward_gap:+.3f} (> 0)"
    )
    print(f"turns kept     {int(main_m['turns'])} over {int(main_m['povs'])} POV-episodes")
    print(f"\nVERDICT: {verdict}  -> {args.out}")
    if verdict == "KILL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

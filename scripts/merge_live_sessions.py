"""Merge the per-segment live-ladder JSONs into the one record a ladder claim is read from.

A ranked-ladder run is played in SEGMENTS because `run_session` has no cross-process resume:
`session.py` counts only its own `log.records`, and `_flush` rewrites `--out` from an empty log on
start, so pointing a second process at the first one's file erases it. Segments also survive a
dropped connection, a killed shell, and an overnight gap. What they do not do is add up by
themselves -- hence this.

THREE THINGS THIS DOES THAT CONCATENATION WOULD NOT.

    ARM AGREEMENT IS CHECKED, NOT ASSUMED. Every segment must agree on the fields that define what
    was played: agent, checkpoint, sample, loop_penalty, team_pool, team_seed, battle_format, mode,
    username and the ladder ack. A mismatch means two different experiments were pooled, which is
    the single most expensive mistake available here and is invisible in the merged numbers.

    THE SUMMARY IS RECOMPUTED, NOT AVERAGED. Rates, the Glicko pair and the Elo fit all come back
    through `telemetry.summarize` over the union of battle records. Averaging per-segment rates
    would silently weight a short segment like a full one, and averaging Glicko is meaningless.

    THE ENDPOINT IS THE PRIMARY READING, AND IT IS NOT DERIVED FROM THE SEGMENTS AT ALL.
    poke-env only ever sees each battle's PRE-battle Elo (telemetry.py's module docstring), so the
    session files cannot contain the final rating by construction. `--rating-json` carries the read
    from https://pokemonshowdown.com/users/<userid>.json, and the merged record keeps the two side
    by side rather than reconciling them.

THE LADDER-DELTA CROSS-CHECK. Showdown scores a disconnect as a LOSS; `summarize` excludes
unfinished battles from every rate. So the endpoint's w+l+t can legitimately exceed the session's
finished count, and the gap is exactly the number of games the account was charged for but the
session did not observe. Reported as `ladder_games_delta`. A large one means the run's own win rate
is optimistic relative to the rating it earned -- which the rating already reflects and the win
rate does not.

    python scripts/merge_live_sessions.py \
      --segment results/live_ladder_gen9ou_seg0.json \
      --segment results/live_ladder_gen9ou_seg1.json \
      --prereg results/live_ladder_gen9ou_prereg.json \
      --rating-json results/live_ladder_gen9ou_rating.json \
      --out results/live_ladder_gen9ou.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rotomai.live.telemetry import BattleRecord, summarize

SCHEMA = "rotomai.live.merged/1"

#: Fields that define the ARM. A disagreement here means the segments are not one experiment.
ARM_FIELDS = (
    "agent",
    "checkpoint",
    "sample",
    "loop_penalty",
    "team_pool",
    "team_seed",
    "battle_format",
    "mode",
    "username",
    "use_live_ratings",
)


class MergeError(RuntimeError):
    """Raised when the segments cannot honestly be pooled."""


def load_segments(paths: list[str]) -> list[dict[str, Any]]:
    out = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise MergeError(f"segment not found: {path}")
        out.append(json.loads(path.read_text()))
    return out


def check_arm_agreement(segments: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    """Return the shared arm, or raise naming the first field that disagrees."""
    if not segments:
        raise MergeError("no segments given")
    arm = {k: segments[0].get(k) for k in ARM_FIELDS}
    for seg, name in zip(segments[1:], names[1:], strict=False):
        for k in ARM_FIELDS:
            if seg.get(k) != arm[k]:
                raise MergeError(
                    f"segments disagree on {k!r}: {names[0]} has {arm[k]!r}, {name} has "
                    f"{seg.get(k)!r}. These are two different experiments and pooling them would "
                    f"produce a number describing neither."
                )
    acks = {json.dumps(s.get("policy", {}).get("ack")) for s in segments}
    if len(acks) > 1:
        raise MergeError(f"segments disagree on the ladder ack: {sorted(acks)}")
    return arm


def dedupe_battles(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union of battle records, keyed by tag. A tag cannot appear twice on a real ladder."""
    seen: dict[str, dict[str, Any]] = {}
    for seg in segments:
        for b in seg.get("battles", []):
            seen.setdefault(b["battle_tag"], b)
    return list(seen.values())


def as_records(battles: list[dict[str, Any]]) -> list[BattleRecord]:
    fields = BattleRecord.__dataclass_fields__
    return [BattleRecord(**{k: v for k, v in b.items() if k in fields}) for b in battles]


def verdict(rating: dict[str, Any] | None, bands: dict[str, Any], n: int) -> dict[str, Any]:
    """Read the PRE-REGISTERED band, or refuse to and say why."""
    floor = 50
    if n < floor:
        return {
            "verdict": "PILOT",
            "reason": f"{n} finished games is under the pre-registered {floor}-game floor; "
            f"no band is read.",
        }
    if not rating:
        return {"verdict": "UNREAD", "reason": "no endpoint rating supplied (--rating-json)"}
    elo, gxe = rating.get("elo"), rating.get("gxe")
    if elo is None or gxe is None:
        return {"verdict": "UNREAD", "reason": "endpoint rating lacks elo/gxe"}
    if elo < 1200 or gxe < 45:
        band = "WEAK"
    elif elo > 1450 or gxe > 60:
        band = "STRONG"
    else:
        band = "DECENT"
    return {
        "verdict": band,
        "reason": f"elo {elo:.1f}, gxe {gxe} against the pre-registered bands",
        "bands": bands,
    }


def merge(
    segments: list[dict[str, Any]],
    names: list[str],
    prereg: dict[str, Any] | None,
    rating: dict[str, Any] | None,
) -> dict[str, Any]:
    arm = check_arm_agreement(segments, names)
    battles = dedupe_battles(segments)
    records = as_records(battles)
    use_live = bool(arm.get("use_live_ratings"))
    summary = summarize(records, use_live_ratings=use_live)

    finished = summary["finished"]
    ladder_games = None
    delta = None
    if rating:
        ladder_games = (rating.get("w") or 0) + (rating.get("l") or 0) + (rating.get("t") or 0)
        delta = ladder_games - finished

    bands = (prereg or {}).get("bands", {})
    merged: dict[str, Any] = {
        "schema": SCHEMA,
        **arm,
        "segments": names,
        "requested_total": sum(s.get("requested_battles", 0) for s in segments),
        "restarts_total": sum(s.get("restarts", 0) for s in segments),
        "policy": segments[0].get("policy"),
        "prereg": (prereg or {}).get("gate"),
        "prereg_n_target": (prereg or {}).get("protocol", {}).get("n_target"),
        **summary,
        "showdown_ladder_final": rating,
        "ladder_games_on_endpoint": ladder_games,
        "ladder_games_delta": delta,
        "ladder_games_delta_note": (
            "endpoint games minus session finished games. Showdown scores a disconnect as a loss "
            "while `summarize` excludes unfinished battles, so a positive delta means the account "
            "was charged for games the session did not observe. The RATING already reflects them; "
            "the session win rate does not."
        ),
        **verdict(rating, bands, finished),
    }
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment", action="append", required=True,
                    help="a live session JSON; pass once per segment, in play order")
    ap.add_argument("--prereg", default=None, help="the pre-registration JSON (bands, n_target)")
    ap.add_argument("--rating-json", default=None,
                    help="endpoint read for the account (scripts/fetch_ladder_rating.py)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    segments = load_segments(args.segment)
    prereg = json.loads(Path(args.prereg).read_text()) if args.prereg else None
    rating = json.loads(Path(args.rating_json).read_text()) if args.rating_json else None
    if rating and "ratings" in rating:  # a raw users.json was passed
        fmt = segments[0].get("battle_format", "gen9ou")
        rating = rating["ratings"].get(fmt)

    merged = merge(segments, list(args.segment), prereg, rating)
    Path(args.out).write_text(json.dumps(merged, indent=2) + "\n")

    print(f"merged {len(segments)} segment(s) -> {args.out}")
    print(f"  finished {merged['finished']}  W/L/T "
          f"{merged['wins']}/{merged['losses']}/{merged['ties']}  "
          f"score_rate {merged['score_rate']}")
    if merged.get("showdown_ladder_final"):
        r = merged["showdown_ladder_final"]
        print(f"  endpoint: elo {r.get('elo')}  gxe {r.get('gxe')}  "
              f"rpr {r.get('rpr')} +/- {r.get('rprd')}")
    if merged.get("ladder_games_delta"):
        print(f"  ladder games delta: {merged['ladder_games_delta']} "
              f"(endpoint counted more games than the session observed)")
    print(f"  VERDICT: {merged['verdict']} -- {merged['reason']}")


if __name__ == "__main__":
    main()

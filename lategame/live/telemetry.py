"""Per-battle records and the Glicko-1/GXE summary for a live session.

Two things here are load-bearing and non-obvious.

THE RATING BACK-FILL IS MANDATORY. poke-env fires ``_battle_finished_callback`` on the ``|win|``
line, but the ``|raw| ...'s rating: ...`` lines are parsed AFTERWARDS in the same message loop.
Reading ``battle.rating`` from the callback therefore always yields ``None``, even on a rated
ladder game. ``LiveLog.finalize`` re-reads every battle before the final write; without it the
rating columns would be silently, uniformly empty and look like "the server never sent them".

WHAT THAT RATING ACTUALLY IS. ``abstract_battle`` parses ``int(rating_info[:4])`` -- the FIRST
number on the raw line, i.e. the rating BEFORE the game -- and Showdown's ladder line reports
**Elo**, not Glicko. So it is recorded as ``showdown_elo_before`` and never enters the Glicko math:
feeding an Elo into a Glicko update is a units error dressed up as precision. (Relatedly,
``battle.rated`` does not exist in poke-env -- "rated" appears only in its ignored-message set --
so ratedness is inferred from the mode, never from the battle object.)
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lategame.eval.rating import MAX_RD, REFERENCE, Rating, gxe, update

SCHEMA = "lategame.live.telemetry/1"

#: Score for a finished battle, by outcome. Ties are 0.5 -- which is exactly why the summary builds
#: its own results list instead of calling ``rate_win_rate`` (see ``summarize``).
_SCORE = {"win": 1.0, "loss": 0.0, "tie": 0.5}


@dataclass
class BattleRecord:
    """One live battle, as observed after it finished (or after we lost the connection)."""

    battle_tag: str
    opponent: str | None
    battle_format: str
    mode: str
    result: str  # win | loss | tie | unfinished
    score: float | None  # None for unfinished -- excluded from every metric
    turns: int
    finished: bool
    showdown_elo_before: int | None = None
    opponent_showdown_elo_before: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    attempt: int = 0


def _classify(battle: Any) -> tuple[str, float | None]:
    """Outcome of a poke-env battle. ``won`` is None on a tie, and False on a loss."""
    if not getattr(battle, "finished", False):
        return "unfinished", None
    won = getattr(battle, "won", None)
    result = "tie" if won is None else ("win" if won else "loss")
    return result, _SCORE[result]


def _elo(value: Any) -> int | None:
    """Best-effort read of poke-env's rating field; a malformed value must never fail a session."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class LiveLog:
    """Accumulates records across restarts, keyed by battle tag.

    Restarts matter: poke-env cannot reconnect, so recovery means building a NEW ``Player`` whose
    ``battles`` dict starts empty. Keying by tag is what merges the pre- and post-restart views
    into one session rather than two.
    """

    def __init__(self) -> None:
        self._records: dict[str, BattleRecord] = {}
        self._seen: dict[str, float] = {}
        self.last_finish_monotonic: float | None = None

    def note_finished(self, battle_tag: str) -> None:
        """Called from poke-env's callback, i.e. on the POKE_LOOP THREAD.

        Deliberately does almost nothing: a dict write and a clock read, both GIL-atomic. All
        record extraction happens on the caller's loop in ``harvest``, so there is no cross-thread
        parsing and no lock.
        """
        self._seen[battle_tag] = time.monotonic()
        self.last_finish_monotonic = self._seen[battle_tag]

    def harvest(self, player: Any, mode: str, attempt: int, battle_format: str) -> None:
        """Fold the player's current battles into the record set. Idempotent by tag."""
        for tag, battle in dict(getattr(player, "battles", {})).items():
            result, score = _classify(battle)
            existing = self._records.get(tag)
            self._records[tag] = BattleRecord(
                battle_tag=tag,
                opponent=getattr(battle, "opponent_username", None),
                battle_format=battle_format,
                mode=mode,
                result=result,
                score=score,
                turns=int(getattr(battle, "turn", 0) or 0),
                finished=bool(getattr(battle, "finished", False)),
                showdown_elo_before=_elo(getattr(battle, "rating", None)),
                opponent_showdown_elo_before=_elo(getattr(battle, "opponent_rating", None)),
                started_at=existing.started_at if existing else None,
                finished_at=existing.finished_at if existing else None,
                duration_s=existing.duration_s if existing else None,
                attempt=existing.attempt if existing else attempt,
            )

    def finalize(self, player: Any, mode: str, attempt: int, battle_format: str) -> None:
        """Re-read every battle before the final write, to back-fill the post-`|win|` rating lines.

        Not an optimisation: at callback time the rating is ALWAYS None (see the module docstring),
        so without this sweep the rating columns are uniformly empty on rated games.
        """
        self.harvest(player, mode, attempt, battle_format)

    @property
    def records(self) -> list[BattleRecord]:
        return list(self._records.values())


def summarize(
    records: Iterable[BattleRecord],
    *,
    use_live_ratings: bool = False,
    opponent_rd: float = MAX_RD,
) -> dict[str, Any]:
    """Win/score rates plus a Glicko-1 rating and GXE for the session.

    UNFINISHED BATTLES ARE EXCLUDED from every rate and from the rating. A battle we dropped
    because our own websocket died is a client artefact; Showdown will likely time it out as a
    loss, but that is evidence about the connection, not about the policy.

    The default (``use_live_ratings=False``) pins every opponent at ``rating.REFERENCE``. That is
    not a fudge: an opponent whose rating we never observed IS the reference player by definition,
    and 1500/350 is exactly what REFERENCE encodes. One session is treated as one Glicko rating
    period.

    Note this builds the results list directly rather than calling ``rate_win_rate``. That helper
    reconstructs wins as ``round(win_rate * n)``, so it can only express scores in {0, 1} and
    CANNOT REPRESENT A TIE. On a tie-free record set the two agree exactly, which is pinned in
    tests.
    """
    records = list(records)
    finished = [r for r in records if r.finished and r.score is not None]
    n = len(finished)

    wins = sum(1 for r in finished if r.result == "win")
    losses = sum(1 for r in finished if r.result == "loss")
    ties = sum(1 for r in finished if r.result == "tie")

    results: list[tuple[Rating, float]] = []
    for r in finished:
        if use_live_ratings and r.opponent_showdown_elo_before is not None:
            opponent = Rating(float(r.opponent_showdown_elo_before), opponent_rd)
        else:
            opponent = REFERENCE
        results.append((opponent, float(r.score or 0.0)))

    rated = update(Rating(), results)

    elos = [r.showdown_elo_before for r in finished if r.showdown_elo_before is not None]
    opp_elos = [
        r.opponent_showdown_elo_before
        for r in finished
        if r.opponent_showdown_elo_before is not None
    ]

    summary: dict[str, Any] = {
        "requested": len(records),
        "finished": n,
        "unfinished": len(records) - n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": round(wins / n, 4) if n else None,
        "score_rate": round((wins + 0.5 * ties) / n, 4) if n else None,
        "mean_turns": round(sum(r.turns for r in finished) / n, 2) if n else None,
        "glicko": round(rated.rating, 1),
        "glicko_rd": round(rated.rd, 1),
        "gxe": round(gxe(rated), 4),
        "gxe_basis": (
            "every opponent pinned at rating.REFERENCE (1500/350); Showdown Elo excluded. Against "
            "a pinned field GXE is a monotone reparameterisation of the score rate plus an "
            "n-driven uncertainty term -- it becomes a real measurement only on a varied field."
            if not use_live_ratings
            else "opponent Showdown ELO approximated as a Glicko rating -- units are not identical"
        ),
        "showdown_elo": {
            "n_reported": len(elos),
            "self_first": elos[0] if elos else None,
            "self_last": elos[-1] if elos else None,
            "opponent_mean": round(sum(opp_elos) / len(opp_elos), 1) if opp_elos else None,
            "note": (
                "PRE-battle Elo: poke-env parses the first number on the |raw| rating line. "
                "Rated ladder games only; None everywhere else is normal."
            ),
        },
    }
    if use_live_ratings:
        summary["opponent_rating_source"] = "showdown_elo_approximated_as_glicko"
    return summary


def write_results(
    path: str | Path,
    cfg_public: Mapping[str, Any],
    summary: Mapping[str, Any],
    records: Iterable[BattleRecord],
) -> None:
    """Write the session file. Rewritten after EVERY battle, so a kill loses at most one game.

    ``cfg_public`` must already be credential-free -- the password is never read into it, and a
    test asserts the serialised bytes do not contain it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        **dict(cfg_public),
        **dict(summary),
        "battles": [asdict(r) for r in records],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)

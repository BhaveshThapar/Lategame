"""Live telemetry: the rating back-fill, the tie case, and the credential guarantee.

The back-fill is the subtle one. poke-env fires its finished-callback on `|win|`, BEFORE the
`|raw| ...'s rating:` line is parsed, so a rating read at callback time is always None. If
`finalize` ever stopped re-reading, rated sessions would silently report empty rating columns and
look exactly like unrated ones.
"""

import json
from types import SimpleNamespace

import pytest

from rotomai.eval.rating import rate_win_rate
from rotomai.live.telemetry import (
    SCHEMA,
    BattleRecord,
    LiveLog,
    summarize,
    write_results,
)


def _battle(tag, *, finished=True, won=True, turn=20, opponent="rival",
            rating=None, opp_rating=None):
    return SimpleNamespace(
        battle_tag=tag, finished=finished, won=won, turn=turn,
        opponent_username=opponent, rating=rating, opponent_rating=opp_rating,
    )


def _player(*battles):
    return SimpleNamespace(battles={b.battle_tag: b for b in battles})


def _rec(result, *, finished=True, turns=10, elo=None, opp_elo=None):
    score = {"win": 1.0, "loss": 0.0, "tie": 0.5}.get(result)
    return BattleRecord(
        battle_tag=f"t-{result}-{turns}-{elo}", opponent="rival", battle_format="gen9ou",
        mode="accept", result=result, score=score, turns=turns, finished=finished,
        showdown_elo_before=elo, opponent_showdown_elo_before=opp_elo,
    )


# --------------------------------------------------------------------------- #
# Classification + harvest
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "finished,won,result,score",
    [(True, True, "win", 1.0), (True, False, "loss", 0.0),
     (True, None, "tie", 0.5), (False, None, "unfinished", None)],
)
def test_classifies_every_outcome_including_ties(finished, won, result, score):
    log = LiveLog()
    log.harvest(_player(_battle("t1", finished=finished, won=won)), "accept", 0, "gen9ou")
    got = log.records[0]
    assert (got.result, got.score) == (result, score)


def test_harvest_is_idempotent_by_tag():
    log = LiveLog()
    player = _player(_battle("t1"), _battle("t2", won=False))
    log.harvest(player, "accept", 0, "gen9ou")
    log.harvest(player, "accept", 0, "gen9ou")
    assert len(log.records) == 2


def test_records_merge_across_a_restart():
    """A rebuilt Player starts with an empty battles dict; the session is still one session."""
    log = LiveLog()
    log.harvest(_player(_battle("before")), "accept", 0, "gen9ou")
    log.harvest(_player(_battle("after", won=False)), "accept", 1, "gen9ou")
    assert {r.battle_tag for r in log.records} == {"before", "after"}


def test_finalize_backfills_a_rating_that_was_none_at_callback_time():
    """THE regression guard: the rating line arrives after `|win|`."""
    log = LiveLog()
    battle = _battle("t1", rating=None, opp_rating=None)
    log.harvest(_player(battle), "ladder", 0, "gen9ou")
    assert log.records[0].showdown_elo_before is None

    battle.rating, battle.opponent_rating = 1043, 1102  # the |raw| lines land
    log.finalize(_player(battle), "ladder", 0, "gen9ou")

    got = log.records[0]
    assert (got.showdown_elo_before, got.opponent_showdown_elo_before) == (1043, 1102)


def test_a_malformed_rating_never_fails_the_session():
    log = LiveLog()
    log.harvest(_player(_battle("t1", rating="not-a-number")), "ladder", 0, "gen9ou")
    assert log.records[0].showdown_elo_before is None


def test_note_finished_tracks_progress_for_the_stall_watchdog():
    log = LiveLog()
    assert log.last_finish_monotonic is None
    log.note_finished("t1")
    assert log.last_finish_monotonic is not None


# --------------------------------------------------------------------------- #
# Summary / GXE
# --------------------------------------------------------------------------- #


def test_matches_rate_win_rate_exactly_on_a_tie_free_record_set():
    """Pins the 'every opponent pinned at REFERENCE' contract against eval/rating.py."""
    records = [_rec("win")] * 10 + [_rec("loss")] * 10
    got = summarize(records)
    expected = rate_win_rate(0.5, 20)
    assert got["glicko"] == round(expected.rating, 1)
    assert got["glicko_rd"] == round(expected.rd, 1)


def test_gxe_orders_all_loss_below_even_below_all_win():
    lose = summarize([_rec("loss")] * 20)["gxe"]
    even = summarize([_rec("win")] * 10 + [_rec("loss")] * 10)["gxe"]
    win = summarize([_rec("win")] * 20)["gxe"]
    assert lose < 0.5 < win
    assert lose < even < win


def test_ties_count_as_half_and_survive_into_the_rating():
    """rate_win_rate cannot express this, which is why summarize builds its own results list."""
    got = summarize([_rec("tie")] * 10)
    assert got["ties"] == 10
    assert got["score_rate"] == 0.5
    assert got["win_rate"] == 0.0  # a tie is not a win
    assert got["glicko"] == pytest.approx(1500.0, abs=1.0)


def test_unfinished_battles_are_excluded_from_every_metric():
    records = [_rec("win")] * 5 + [_rec("unfinished", finished=False)] * 3
    got = summarize(records)
    assert (got["requested"], got["finished"], got["unfinished"]) == (8, 5, 3)
    assert got["win_rate"] == 1.0  # the dropped battles do not dilute it
    assert got["wins"] == 5


def test_empty_session_does_not_divide_by_zero():
    """player.win_rate would raise ZeroDivisionError here; ours must not."""
    got = summarize([])
    assert got["win_rate"] is None and got["score_rate"] is None and got["mean_turns"] is None
    assert got["finished"] == 0


def test_gxe_basis_states_the_degeneracy():
    basis = summarize([_rec("win")])["gxe_basis"]
    assert "REFERENCE" in basis and "monotone reparameterisation" in basis


def test_showdown_elo_is_reported_separately_and_never_enters_glicko():
    with_elo = summarize([_rec("win", elo=1200, opp_elo=1300)] * 4)
    without = summarize([_rec("win")] * 4)
    assert with_elo["glicko"] == without["glicko"]  # Elo must not move the Glicko rating
    assert with_elo["showdown_elo"]["n_reported"] == 4
    assert with_elo["showdown_elo"]["opponent_mean"] == 1300.0
    assert without["showdown_elo"]["n_reported"] == 0


def test_use_live_ratings_summarises_on_the_elo_scale_not_the_glicko_one():
    """Elo and Glicko share the 400-point logistic and NOTHING else. Feeding an observed ladder
    Elo into `Rating` read every opponent as ~500 points weaker than it was: a real 7-18 session
    against mean opponent Elo 1045 reported Glicko 501.4 / GXE 0.0242 while the account's own page
    said Elo 1017 / GXE 30%. The Glicko half must therefore be scale-pure -- identical with the
    flag on or off -- and the observed ratings summarised on their own scale."""
    live = summarize([_rec("win", opp_elo=1800)] * 3 + [_rec("loss", opp_elo=1800)] * 2,
                     use_live_ratings=True)
    pinned = summarize([_rec("win")] * 3 + [_rec("loss")] * 2)

    assert live["opponent_rating_source"] == "showdown_elo_fitted_on_the_elo_scale"
    # The Glicko/GXE pair is the PINNED-field one either way; observed Elo never enters it.
    assert live["glicko"] == pinned["glicko"]
    assert live["gxe"] == pinned["gxe"]
    # ...and the observed opponents are summarised on the Elo scale instead.
    e = live["showdown_elo"]
    assert e["elo_mle_bounded"] is True
    assert e["elo_mle"] > 1800  # won 3 of 5 against 1800s, so it fits above them
    assert 0.0 < e["expected_score_vs_mean_opponent"] < 1.0


def test_an_all_wins_record_has_no_finite_elo_and_says_so():
    """The likelihood is monotone with no losses, so the MLE is infinite. Emitting the clamp as if
    it were a measurement is exactly the kind of confident-wrong number this file exists to stop."""
    got = summarize([_rec("win", opp_elo=1200)] * 4, use_live_ratings=True)
    assert got["showdown_elo"]["elo_mle_bounded"] is False
    assert got["showdown_elo"]["elo_mle"] is None
    assert got["showdown_elo"]["expected_score_vs_mean_opponent"] is None


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #


def test_results_file_is_valid_and_carries_the_schema(tmp_path):
    out = tmp_path / "live.json"
    records = [_rec("win"), _rec("loss")]
    write_results(out, {"agent": "ppo", "mode": "accept"}, summarize(records), records)

    got = json.loads(out.read_text())
    assert got["schema"] == SCHEMA
    assert got["agent"] == "ppo"
    assert len(got["battles"]) == 2
    assert got["battles"][0]["battle_tag"]


def test_no_temp_file_is_left_behind(tmp_path):
    out = tmp_path / "live.json"
    write_results(out, {}, summarize([]), [])
    assert list(tmp_path.iterdir()) == [out]


def test_the_password_can_never_reach_the_results_file(tmp_path):
    """cfg_public is built without the secret; this pins that the file stays clean."""
    secret = "hunter2-do-not-leak"
    out = tmp_path / "live.json"
    write_results(out, {"username": "bot", "server": "wss://sim3.psim.us/showdown/websocket"},
                  summarize([_rec("win")]), [_rec("win")])
    raw = out.read_bytes()
    assert secret.encode() not in raw
    assert b"ROTOMAI_PS_PASSWORD" not in raw

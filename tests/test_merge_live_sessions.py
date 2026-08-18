"""Pooling ladder segments: the checks, not the arithmetic.

Concatenating segments is trivial; the parts worth testing are the ones that REFUSE to concatenate,
because a silently-wrong pool here produces a published rating describing an experiment nobody ran.
No network and no server: every case is built from literals.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from merge_live_sessions import (  # noqa: E402
    MergeError,
    check_arm_agreement,
    dedupe_battles,
    merge,
    verdict,
)

_ARM = {
    "agent": "offrl",
    "checkpoint": "checkpoints/ppo_v26b_s0/iter_320.pt",
    "sample": False,
    "loop_penalty": 4.0,
    "team_pool": "rotomai/teambuilding/data/teams_gen9ou.packed",
    "team_seed": 0,
    "battle_format": "gen9ou",
    "mode": "ladder",
    "username": "RotomLover12",
    "use_live_ratings": True,
}


def _battle(tag, result="win", elo=1040):
    score = {"win": 1.0, "loss": 0.0, "tie": 0.5}[result]
    return {
        "battle_tag": tag, "opponent": "rival", "battle_format": "gen9ou", "mode": "ladder",
        "result": result, "score": score, "turns": 20, "finished": True,
        "showdown_elo_before": 1000, "opponent_showdown_elo_before": elo,
    }


def _seg(battles, **over):
    d = dict(_ARM)
    d.update({"battles": battles, "requested_battles": len(battles), "restarts": 0,
              "policy": {"ranked": True, "ack": "i-have-read-plan-md-section-15"}})
    d.update(over)
    return d


# --------------------------------------------------------------------------- #
# The refusals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "field,bad",
    [
        ("checkpoint", "checkpoints/ppo_v25b_s0/iter_160.pt"),
        ("loop_penalty", 0.0),
        ("team_seed", 7),
        ("agent", "bc"),
        ("sample", True),
        ("username", "SomeoneElse"),
    ],
)
def test_segments_from_different_experiments_are_refused(field, bad):
    """Each of these silently changes what was played. Pooling across one produces a number that
    describes neither arm, and nothing downstream could detect it."""
    a = _seg([_battle("a")])
    b = _seg([_battle("b")], **{field: bad})
    with pytest.raises(MergeError, match=field):
        check_arm_agreement([a, b], ["a.json", "b.json"])


def test_a_disagreeing_ladder_ack_is_refused():
    a = _seg([_battle("a")])
    b = _seg([_battle("b")], policy={"ranked": True, "ack": None})
    with pytest.raises(MergeError, match="ack"):
        check_arm_agreement([a, b], ["a.json", "b.json"])


def test_matching_segments_pass_and_return_the_shared_arm():
    arm = check_arm_agreement([_seg([_battle("a")]), _seg([_battle("b")])], ["a", "b"])
    assert arm["checkpoint"] == _ARM["checkpoint"]
    assert arm["loop_penalty"] == 4.0


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #


def test_a_battle_seen_in_two_segments_is_counted_once():
    """Segments can overlap if one was restarted against a stale --out. A battle tag is unique on
    Showdown, so the tag is the identity and double-counting it would inflate n."""
    a = _seg([_battle("shared"), _battle("only-a")])
    b = _seg([_battle("shared"), _battle("only-b")])
    assert len(dedupe_battles([a, b])) == 3


def test_the_summary_is_recomputed_over_the_union_not_averaged():
    """A 1-game segment must not weigh the same as a 9-game one."""
    big = _seg([_battle(f"w{i}", "win") for i in range(9)])
    small = _seg([_battle("l0", "loss")])
    out = merge([big, small], ["big", "small"], None, None)
    assert out["finished"] == 10
    assert out["score_rate"] == 0.9  # not the 0.5 an average of segment rates would give


# --------------------------------------------------------------------------- #
# The pre-registered verdict
# --------------------------------------------------------------------------- #


def test_under_the_floor_no_band_is_read():
    """The pre-registration says fewer than 50 finished games is a pilot. Reading a band off 25
    games would be exactly the after-the-fact choice pre-registration exists to prevent."""
    got = verdict({"elo": 1017.0, "gxe": 30}, {}, n=25)
    assert got["verdict"] == "PILOT"
    assert "floor" in got["reason"]


@pytest.mark.parametrize(
    "elo,gxe,expected",
    [
        (1017.0, 30, "WEAK"),
        (1199.0, 44, "WEAK"),
        (1300.0, 50, "DECENT"),
        (1450.0, 60, "DECENT"),
        (1600.0, 70, "STRONG"),
    ],
)
def test_the_bands_are_read_exactly_as_pre_registered(elo, gxe, expected):
    assert verdict({"elo": elo, "gxe": gxe}, {}, n=100)["verdict"] == expected


def test_a_missing_endpoint_read_is_unread_rather_than_guessed():
    assert verdict(None, {}, n=100)["verdict"] == "UNREAD"


# --------------------------------------------------------------------------- #
# The cross-check that makes a disconnect visible
# --------------------------------------------------------------------------- #


def test_games_the_session_never_saw_show_up_as_a_delta():
    """Showdown scores a disconnect as a loss; `summarize` drops unfinished battles. The gap is
    real and must be surfaced, not reconciled away -- the RATING already paid for those games."""
    seg = _seg([_battle(f"b{i}", "win") for i in range(10)])
    out = merge([seg], ["seg"], None, {"elo": 1100.0, "gxe": 40, "w": 10, "l": 3, "t": 0})
    assert out["finished"] == 10
    assert out["ladder_games_on_endpoint"] == 13
    assert out["ladder_games_delta"] == 3


def test_the_merged_record_round_trips_as_json(tmp_path):
    out = merge([_seg([_battle("a"), _battle("b", "loss")])], ["seg"], None, None)
    p = tmp_path / "merged.json"
    p.write_text(json.dumps(out, indent=2))
    assert json.loads(p.read_text())["finished"] == 2

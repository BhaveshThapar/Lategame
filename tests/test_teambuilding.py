"""Unit tests for R-TEAM team provisioning (no server, no node).

TeamPool is a poke-env Teambuilder that yields a random packed team per battle; these tests
pin its sampling contract (in-pool, seeded-deterministic, varied) and the paste->packed helper.
"""

import pytest

from rotomai.teambuilding.pool import TeamPool, pack_showdown_team

_PASTE = """\
Great Tusk @ Leftovers
Ability: Protosynthesis
Tera Type: Steel
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Headlong Rush
- Close Combat
- Rapid Spin
- Stealth Rock
"""


def test_yield_team_is_in_pool():
    teams = ["a", "b", "c", "d"]
    pool = TeamPool(teams, seed=0)
    draws = {pool.yield_team() for _ in range(100)}
    assert draws <= set(teams)
    assert len(draws) > 1  # a pool of >1 must not collapse to a single team


def test_yield_team_is_seed_deterministic():
    teams = ["t1", "t2", "t3", "t4", "t5"]
    p1, p2 = TeamPool(teams, seed=7), TeamPool(teams, seed=7)
    assert [p1.yield_team() for _ in range(20)] == [p2.yield_team() for _ in range(20)]
    # Different seeds must not lock-step -- the sweep gives p1/p2 distinct seeds for
    # independent (non-mirror) team draws.
    q = TeamPool(teams, seed=8)
    fresh = TeamPool(teams, seed=7)
    assert [fresh.yield_team() for _ in range(20)] != [q.yield_team() for _ in range(20)]


def test_empty_pool_rejected():
    with pytest.raises(ValueError):
        TeamPool([])


def test_from_packed_file_ignores_blank_lines(tmp_path):
    f = tmp_path / "pool.packed"
    f.write_text("teamA\n\nteamB\n  \nteamC\n")
    pool = TeamPool.from_packed_file(f, seed=0)
    assert pool.teams == ["teamA", "teamB", "teamC"]
    assert pool.yield_team() in {"teamA", "teamB", "teamC"}


def test_from_packed_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TeamPool.from_packed_file(tmp_path / "nope.packed")


def test_pack_showdown_team_produces_packed_format():
    packed = pack_showdown_team(_PASTE)
    assert "|" in packed  # packed uses '|' field separators
    assert "Great Tusk" in packed
    assert "headlongrush" in packed  # moves are id-normalised in packed form

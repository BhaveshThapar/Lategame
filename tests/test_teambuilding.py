"""Unit tests for R-TEAM team provisioning (no server, no node).

TeamPool is a poke-env Teambuilder that yields a random packed team per battle; these tests
pin its sampling contract (in-pool, seeded-deterministic, varied) and the paste->packed helper.
"""

from pathlib import Path

import pytest

from rotomai.teambuilding.pool import (
    DEFAULT_OU_POOL,
    DEFAULT_VGC_POOL,
    TeamPool,
    pack_showdown_team,
    resolve_pool_path,
)

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


# --------------------------------------------------------------------------- #
# CWD independence: a standing live service is not launched from the checkout
# --------------------------------------------------------------------------- #


def test_the_default_pools_stay_repo_relative():
    """Their string form is written into every results file as `team_pool`, and
    `merge_live_sessions.py` refuses to pool segments whose arm fields disagree. Absolutising them
    would make the same arm on two machines look like two experiments."""
    assert not DEFAULT_OU_POOL.is_absolute()
    assert not DEFAULT_VGC_POOL.is_absolute()
    assert str(DEFAULT_OU_POOL) == "rotomai/teambuilding/data/teams_gen9ou.packed"


@pytest.mark.parametrize("pool", [DEFAULT_OU_POOL, DEFAULT_VGC_POOL])
def test_a_default_pool_loads_from_any_working_directory(pool, tmp_path, monkeypatch):
    """Before this, `rotomai live` died with a bare FileNotFoundError anywhere but the repo root --
    which is every systemd unit and every container."""
    monkeypatch.chdir(tmp_path)
    assert not Path(pool).exists(), "the fixture is only meaningful outside the checkout"
    loaded = TeamPool.from_packed_file(pool)
    assert loaded.teams


def test_resolution_prefers_a_real_relative_path_over_the_package_copy(tmp_path, monkeypatch):
    """A caller who points at their own file must get their own file, not the shipped pool."""
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "teams_gen9ou.packed"
    local.write_text("mine|||||||||\n")
    assert resolve_pool_path("teams_gen9ou.packed") == Path("teams_gen9ou.packed")
    assert TeamPool.from_packed_file("teams_gen9ou.packed").teams == ["mine|||||||||"]


def test_an_unknown_name_is_returned_unchanged_so_the_error_names_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_pool_path("nope.packed") == Path("nope.packed")
    with pytest.raises(FileNotFoundError, match="nope.packed"):
        TeamPool.from_packed_file("nope.packed")

"""Unit tests for the shaped state-value reward (no server, no torch)."""

from types import SimpleNamespace

import pytest

from lategame.data.reward import RewardWeights, state_value

W = RewardWeights()


def _mon(hp=1.0, fainted=False, status=None):
    return SimpleNamespace(current_hp_fraction=hp, fainted=fainted, status=status)


def _battle(team, opp, won=None, lost=None):
    return SimpleNamespace(
        team={str(i): m for i, m in enumerate(team)},
        opponent_team={str(i): m for i, m in enumerate(opp)},
        won=won,
        lost=lost,
    )


def test_symmetric_full_teams_value_zero():
    b = _battle([_mon() for _ in range(6)], [_mon() for _ in range(6)])
    assert state_value(b, W) == pytest.approx(0.0)


def test_win_minus_loss_is_two_victory_values():
    team, opp = [_mon() for _ in range(6)], [_mon() for _ in range(6)]
    won = state_value(_battle(team, opp, won=True), W)
    lost = state_value(_battle(team, opp, lost=True), W)
    assert won - lost == 2 * W.victory_value


def test_own_faint_lowers_value_opponent_faint_raises_it():
    base = _battle([_mon() for _ in range(6)], [_mon() for _ in range(6)])
    own_faint = _battle(
        [_mon(hp=0.0, fainted=True)] + [_mon() for _ in range(5)], [_mon() for _ in range(6)]
    )
    opp_faint = _battle(
        [_mon() for _ in range(6)], [_mon(hp=0.0, fainted=True)] + [_mon() for _ in range(5)]
    )
    assert state_value(own_faint, W) < state_value(base, W) < state_value(opp_faint, W)


def test_faint_is_symmetric_between_sides():
    own_faint = _battle(
        [_mon(hp=0.0, fainted=True)] + [_mon() for _ in range(5)], [_mon() for _ in range(6)]
    )
    opp_faint = _battle(
        [_mon() for _ in range(6)], [_mon(hp=0.0, fainted=True)] + [_mon() for _ in range(5)]
    )
    assert state_value(own_faint, W) == pytest.approx(-state_value(opp_faint, W))


def test_sparse_weights_zero_interior_pm_one_terminal():
    sparse = RewardWeights(hp_value=0.0, fainted_value=0.0, status_value=0.0, victory_value=1.0)
    interior = _battle([_mon(hp=0.5)], [_mon(fainted=True)])
    assert state_value(interior, sparse) == 0.0
    assert state_value(_battle([_mon()], [_mon()], won=True), sparse) == 1.0
    assert state_value(_battle([_mon()], [_mon()], lost=True), sparse) == -1.0


# --------------------------------------------------------------------------- #
# The digest mirror (Build 28 B5b)
# --------------------------------------------------------------------------- #
def _digest_from(battle) -> dict:
    """The driver's digest shape, built from a poke-env battle in the DRIVER frame (p1 = us)."""

    def side(mons) -> dict:
        return {
            m.species: {
                "hp": round(0.0 if m.fainted else float(m.current_hp_fraction or 0.0), 2),
                "status": "" if (m.status is None or m.status.name == "FNT") else m.status.name,
                "fainted": bool(m.fainted),
                "active": False,
                "boosts": {},
            }
            for m in mons
        }

    return {"p1": side(battle.team.values()), "p2": side(battle.opponent_team.values())}


def test_the_digest_value_mirrors_state_value():
    """These are two implementations of one formula over different inputs, so they must agree --
    the digest one exists only because scoring a search leaf through poke-env costs ~170 ms of
    deepcopy + delta replay against ~1 ms for the RPC that already produced the digest."""
    from poke_env.battle import Pokemon, Status

    from lategame.data.reward import RewardWeights, digest_state_value, state_value

    class _B:
        def __init__(self, ours, theirs, won=None):
            self.team = {m.species: m for m in ours}
            self.opponent_team = {m.species: m for m in theirs}
            self.won = won is True
            self.lost = won is False

    def mon(species, hp=1.0, fainted=False, status=None):
        m = Pokemon(9, species=species)
        # A fainted Pokemon is at 0 HP in poke-env AND in the simulator the digest reads, so the
        # fixture must not model one at full HP -- that would make the two sides disagree for a
        # reason the real code never encounters.
        m._current_hp = 0 if fainted else int(hp * 100)
        m._max_hp = 100
        if fainted:
            m._status = Status.FNT
        elif status is not None:
            m._status = status
        return m

    w = RewardWeights()
    for ours, theirs, won in [
        ([mon("greattusk"), mon("kingambit", hp=0.4)], [mon("dragonite", hp=0.7)], None),
        ([mon("greattusk", fainted=True)], [mon("dragonite", status=Status.BRN)], None),
        ([mon("greattusk")], [mon("dragonite")], True),
        ([mon("greattusk")], [mon("dragonite")], False),
    ]:
        b = _B(ours, theirs, won)
        winner = "p1" if won is True else "p2" if won is False else None
        expected = state_value(b, w)
        actual = digest_state_value(_digest_from(b), w, winner=winner)
        assert actual == pytest.approx(expected, abs=1e-6), (expected, actual)


def test_the_digest_value_credits_unrevealed_mons_the_same_way():
    """`state_value` credits the mons it has not seen at full HP; dropping that on the digest side
    would make a leaf drift as the opponent's team is revealed, which is a moving target."""
    from lategame.data.reward import RewardWeights, digest_state_value

    w = RewardWeights()
    full = {f"m{i}": {"hp": 1.0, "status": "", "fainted": False} for i in range(6)}
    one = {"m0": {"hp": 1.0, "status": "", "fainted": False}}
    # Six known vs one known + five credited: identical, because the credit is full HP.
    assert digest_state_value({"p1": full, "p2": full}, w) == pytest.approx(
        digest_state_value({"p1": one, "p2": one}, w), abs=1e-9
    )

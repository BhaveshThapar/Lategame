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

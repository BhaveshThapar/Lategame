"""Unit tests for the R-CALC seed move-scoring helpers (no server needed)."""

from poke_env.battle import MoveCategory, PokemonType

from rotomai.engine.damage import (
    expected_power,
    normalize_accuracy,
    score_move,
    stab_multiplier,
)


class FakeMove:
    def __init__(self, base_power, move_type, accuracy=1.0, category=MoveCategory.PHYSICAL):
        self.base_power = base_power
        self.type = move_type
        self.accuracy = accuracy
        self.category = category


class FakeAttacker:
    def __init__(self, types):
        self.types = types


class FakeDefender:
    def __init__(self, multiplier):
        self._multiplier = multiplier

    def damage_multiplier(self, _move):
        return self._multiplier


def test_stab_multiplier_matches_attacker_type():
    assert stab_multiplier(PokemonType.FIRE, (PokemonType.FIRE, PokemonType.FLYING)) == 1.5
    assert stab_multiplier(PokemonType.FIRE, (PokemonType.WATER, None)) == 1.0
    assert stab_multiplier(None, (PokemonType.FIRE,)) == 1.0


def test_normalize_accuracy_handles_never_miss():
    assert normalize_accuracy(True) == 1.0
    assert normalize_accuracy(None) == 1.0
    assert normalize_accuracy(0.9) == 0.9


def test_expected_power_is_product():
    assert expected_power(100, 1.5, 2.0, 1.0) == 300.0


def test_score_move_combines_stab_and_effectiveness():
    move = FakeMove(base_power=100, move_type=PokemonType.FIRE)
    attacker = FakeAttacker(types=(PokemonType.FIRE, None))
    defender = FakeDefender(multiplier=2.0)  # super effective
    assert score_move(move, attacker, defender) == 100 * 1.5 * 2.0 * 1.0


def test_status_and_zero_power_moves_score_zero():
    attacker = FakeAttacker(types=(PokemonType.FIRE,))
    defender = FakeDefender(multiplier=1.0)
    status = FakeMove(base_power=0, move_type=PokemonType.NORMAL, category=MoveCategory.STATUS)
    zero_power = FakeMove(base_power=0, move_type=PokemonType.FIRE)
    assert score_move(status, attacker, defender) == 0.0
    assert score_move(zero_power, attacker, defender) == 0.0


def test_score_move_with_no_defender_assumes_neutral():
    move = FakeMove(base_power=80, move_type=PokemonType.WATER, accuracy=0.9)
    attacker = FakeAttacker(types=(PokemonType.WATER,))
    assert score_move(move, attacker, None) == 80 * 1.5 * 1.0 * 0.9

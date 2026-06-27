"""Unit tests for the feature encoder (no server needed).

Uses lightweight fakes (in the style of test_damage.py) for the per-entity
encoders, and a minimal empty fake battle to exercise ``embed_battle``'s
assembly + padding + the fixed-size assertion.
"""

import numpy as np
from poke_env.battle import PokemonType, Status

from lategame.features import encoder
from lategame.features.encoder import (
    _MOVE_DIM,
    _POKEMON_DIM,
    OBS_DIM,
    _encode_move,
    _encode_pokemon,
    _one_hot,
    _presence,
    embed_battle,
)


class FakePokemon:
    def __init__(self, types, hp=1.0, active=True, fainted=False, status=None):
        self.types = types
        self.current_hp_fraction = hp
        self.active = active
        self.fainted = fainted
        self.status = status
        self.boosts = {k: 0 for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")}
        self.base_stats = {k: 100 for k in ("hp", "atk", "def", "spa", "spd", "spe")}
        self.moves = {}


class FakeBattle:
    def __init__(self):
        self.team = {}
        self.opponent_team = {}
        self.active_pokemon = None
        self.opponent_active_pokemon = None
        self.weather = {}
        self.fields = {}
        self.side_conditions = {}
        self.opponent_side_conditions = {}
        self.turn = 0
        self.force_switch = False
        self.can_tera = False
        self.maybe_trapped = False


def test_one_hot():
    assert _one_hot(2, 4).tolist() == [0.0, 0.0, 1.0, 0.0]
    assert _one_hot(None, 3).tolist() == [0.0, 0.0, 0.0]


def test_presence_stackable_vs_boolean():
    from poke_env.battle import SideCondition

    members = [SideCondition.SPIKES, SideCondition.STEALTH_ROCK]
    vec = _presence({SideCondition.SPIKES: 3, SideCondition.STEALTH_ROCK: 7}, members)
    assert vec[0] == 1.0  # 3/3 layers
    assert vec[1] == 1.0  # boolean presence, not the raw turn value


def test_encode_missing_entities_are_zero():
    assert _encode_pokemon(None).shape == (_POKEMON_DIM,)
    assert not _encode_pokemon(None).any()
    assert _encode_move(None, None, None).shape == (_MOVE_DIM,)
    assert not _encode_move(None, None, None).any()


def test_encode_pokemon_sets_type_and_hp():
    mon = FakePokemon(types=[PokemonType.FIRE], hp=0.5, status=Status.BRN)
    vec = _encode_pokemon(mon)
    assert vec.shape == (_POKEMON_DIM,)
    assert vec[0] == 1.0  # present
    assert vec[3] == 0.5  # hp fraction
    fire_idx = 4 + encoder._TYPE_INDEX[PokemonType.FIRE]
    assert vec[fire_idx] == 1.0


def test_embed_battle_is_fixed_size_float32():
    obs = embed_battle(FakeBattle())
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_embed_battle_is_deterministic():
    battle = FakeBattle()
    np.testing.assert_array_equal(embed_battle(battle), embed_battle(battle))

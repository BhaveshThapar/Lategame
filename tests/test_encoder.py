"""Unit tests for the feature encoder (no server needed).

Uses lightweight fakes (in the style of test_damage.py) for the per-entity
encoders, and a minimal empty fake battle to exercise ``embed_battle``'s
assembly + padding + the fixed-size assertion.
"""

import numpy as np
from poke_env.battle import PokemonType, Status

from rotomai.features import encoder
from rotomai.features.encoder import (
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
    def __init__(
        self, types, hp=1.0, active=True, fainted=False, status=None,
        species=None, item=None, ability=None, first_turn=False,
    ):
        self.types = types
        self.current_hp_fraction = hp
        self.active = active
        self.fainted = fainted
        self.status = status
        self.first_turn = first_turn
        self.species = species  # id-str or None (-> UNK)
        self.item = item
        self.ability = ability
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


def test_encode_pokemon_appends_identity_ids():
    from rotomai.features import vocab

    v = vocab.load_vocab()
    mon = FakePokemon(
        types=[PokemonType.GROUND],
        species="greattusk",
        item="leftovers",
        ability="protosynthesis",
    )
    vec = _encode_pokemon(mon)
    # POKEMON_ID_FIELDS order: species, item, ability -- the trailing 3 channels.
    assert vec[-3] == v.index("species", "greattusk") > 0
    assert vec[-2] == v.index("items", "leftovers") > 0
    assert vec[-1] == v.index("abilities", "protosynthesis") > 0
    # Unknown/None identity -> UNK (0).
    bare = FakePokemon(types=[PokemonType.GROUND])
    assert list(_encode_pokemon(bare)[-3:]) == [0.0, 0.0, 0.0]


def test_embed_battle_is_fixed_size_float32():
    obs = embed_battle(FakeBattle())
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_embed_battle_is_deterministic():
    battle = FakeBattle()
    np.testing.assert_array_equal(embed_battle(battle), embed_battle(battle))


def test_obs_version_v5_marks_first_turn_feature():
    # Build 9 Gate B bump: rejects v4-era shards/checkpoints.
    assert encoder.OBS_VERSION.startswith("v5-")


def test_embed_battle_move_blocks_canonical_and_insertion_invariant():
    from poke_env.battle import Move

    from rotomai.features import vocab
    from rotomai.features.encoder import OBS_LAYOUT

    def build(insertion_order):
        battle = FakeBattle()
        mon = FakePokemon(types=[PokemonType.ELECTRIC], species="pikachu")
        mon.moves = {mid: Move(mid, gen=9) for mid in insertion_order}
        battle.team = {"p1: Pikachu": mon}
        battle.active_pokemon = mon  # no opponent: score_move handles a None defender
        return embed_battle(battle)

    reveal = build(["voltswitch", "surf"])  # offline: reveal/backfill insertion order
    declared = build(["surf", "voltswitch"])  # live: request declaration order
    np.testing.assert_array_equal(reveal, declared)  # obs is a function of the move SET
    ms, md = OBS_LAYOUT.moves_start, OBS_LAYOUT.move_dim
    v = vocab.load_vocab()
    assert reveal[ms + md - 1] == v.index("moves", "surf")  # canonical slot 0
    assert reveal[ms + md + md - 1] == v.index("moves", "voltswitch")  # canonical slot 1


def test_global_encodes_own_active_first_turn():
    """Build 9 Gate B: the last global scalar is own active.first_turn (recency flag)."""
    ft_idx = OBS_DIM - 1  # global block is last; first_turn is the last of the 5 global scalars

    battle = FakeBattle()
    just_switched = FakePokemon(types=[PokemonType.ELECTRIC], first_turn=True)
    battle.team = {"p1: Pikachu": just_switched}
    battle.active_pokemon = just_switched
    assert embed_battle(battle)[ft_idx] == 1.0

    battle.active_pokemon = FakePokemon(types=[PokemonType.ELECTRIC], first_turn=False)
    assert embed_battle(battle)[ft_idx] == 0.0

    # No active mon (team preview / forced switch) -> 0.0, no AttributeError.
    assert embed_battle(FakeBattle())[ft_idx] == 0.0

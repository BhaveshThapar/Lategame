"""Unit tests for the canonical action codec (no server needed).

Build 5 stopped delegating to poke-env's ``SinglesEnv``: move slots 6-9 now index
``canonical_moves`` (known moves sorted by id, first four) so slot semantics are identical
for a log-reconstructed battle (reveal + backfill insertion order) and a live request-backed
battle (declaration insertion order). These tests pin the canonical ordering, the
label/decode/mask alignment on a real request-backed ``Battle``, the round-trip identity on
every masked-legal action, and the preserved poke-env lone-available-move fallback.
"""

from __future__ import annotations

import logging

from poke_env.battle import Battle, Move, Pokemon
from poke_env.player import SingleBattleOrder

from lategame.features import vocab
from lategame.features.action_space import (
    GEN9_ACTION_SPACE_SIZE,
    action_mask,
    action_space_size,
    action_to_order,
    canonical_moves,
    label_action,
    order_to_action,
)
from lategame.features.encoder import OBS_LAYOUT, embed_battle

_LOGGER = logging.getLogger("test_action_space")

# Declaration (request) order is deliberately non-canonical: sorted -> irontail, surf,
# thunderbolt, voltswitch, so every declared slot differs from its canonical slot.
_DECLARED = ["voltswitch", "thunderbolt", "irontail", "surf"]
_CANONICAL = sorted(_DECLARED)


class FakeGenBattle:
    def __init__(self, gen):
        self.gen = gen


def _req_move(mid: str, disabled: bool = False) -> dict:
    return {
        "move": mid.title(), "id": mid, "pp": 24, "maxpp": 24,
        "target": "normal", "disabled": disabled,
    }


def _req_mon(ident: str, details: str, active: bool, moves: list[str]) -> dict:
    return {
        "ident": ident,
        "details": details,
        "condition": "100/100",
        "active": active,
        "stats": {"atk": 120, "def": 120, "spa": 150, "spd": 120, "spe": 150},
        "moves": moves,
        "baseAbility": "static",
        "item": "lightball",
        "pokeball": "pokeball",
        "ability": "static",
        "commanding": False,
        "reviving": False,
        "teraType": "Electric",
        "terastallized": "",
    }


def _live_battle(disabled: set[str] = frozenset(), lone_move: str | None = None) -> Battle:
    """A request-backed battle: Pikachu active (moves declared non-canonically), Eevee bench.

    ``lone_move`` replaces the active request block's move list with that single move (the
    forced Struggle/locked-move state) while the side's known moveset stays the full four.
    """
    battle = Battle("battle-gen9ou-t", "Alice", _LOGGER, gen=9)
    for line in (
        "|player|p1|Alice|1|",
        "|player|p2|Bob|2|",
        "|switch|p1a: Pikachu|Pikachu, L84, M|100/100",
        "|switch|p2a: Bulbasaur|Bulbasaur, L88, M|100/100",
    ):
        battle.parse_message(line.split("|"))
    active_moves = (
        [_req_move(lone_move)]
        if lone_move
        else [_req_move(m, disabled=m in disabled) for m in _DECLARED]
    )
    battle.parse_request(
        {
            "active": [{"moves": active_moves, "canTerastallize": "Electric"}],
            "side": {
                "name": "Alice",
                "id": "p1",
                "pokemon": [
                    _req_mon("p1: Pikachu", "Pikachu, L84, M", True, _DECLARED),
                    _req_mon("p1: Eevee", "Eevee, L84, F", False, ["quickattack"]),
                ],
            },
        }
    )
    return battle


def test_gen9_action_space_size():
    assert GEN9_ACTION_SPACE_SIZE == 26  # 6 switches + 4 moves * (4 gimmicks + 1)


def test_action_space_size_tracks_generation():
    assert action_space_size(FakeGenBattle(gen=9)) == 26
    assert action_space_size(FakeGenBattle(gen=1)) == 10  # no gimmicks


def test_canonical_moves_sorted_and_insertion_invariant():
    a = Pokemon(gen=9, species="pikachu")
    b = Pokemon(gen=9, species="pikachu")
    for mid in _DECLARED:
        a._add_move(mid)
    for mid in reversed(_DECLARED):
        b._add_move(mid)
    assert [m.id for m in canonical_moves(a)] == _CANONICAL
    assert [m.id for m in canonical_moves(a)] == [m.id for m in canonical_moves(b)]


def test_canonical_moves_keeps_four_lexicographically_smallest():
    mon = Pokemon(gen=9, species="ditto")
    for mid in ("tackle", "surf", "agility", "voltswitch", "bite"):
        mon._add_move(mid)
    assert [m.id for m in canonical_moves(mon)] == ["agility", "bite", "surf", "tackle"]


def test_label_action_uses_canonical_slots_from_public_state():
    battle = Battle("battle-gen9rb-t", "Alice", _LOGGER, gen=9)
    for line in (
        "|player|p1|Alice|1|",
        "|player|p2|Bob|2|",
        "|switch|p1a: Pikachu|Pikachu, L84, M|100/100",
        "|switch|p2a: Bulbasaur|Bulbasaur, L88, M|100/100",
        # Anti-canonical reveal order: Volt Switch first, Surf second.
        "|move|p1a: Pikachu|Volt Switch|p2a: Bulbasaur",
        "|move|p1a: Pikachu|Surf|p2a: Bulbasaur",
    ):
        battle.parse_message(line.split("|"))
    active = battle.active_pokemon
    assert list(active.moves) == ["voltswitch", "surf"]  # insertion = reveal order
    # Canonical slots: surf 0 -> action 6, voltswitch 1 -> action 7 (reveal order says 7/6).
    assert label_action(SingleBattleOrder(active.moves["surf"]), battle) == 6
    assert label_action(SingleBattleOrder(active.moves["voltswitch"]), battle) == 7
    # A move outside the known set is un-indexable -> negative, caller drops the turn.
    assert label_action(SingleBattleOrder(Move("thunderbolt", gen=9)), battle) == -2


def test_decode_uses_canonical_not_declaration_order():
    battle = _live_battle()
    decoded = [action_to_order(6 + j, battle).order.id for j in range(4)]
    assert decoded == _CANONICAL  # slot j = j-th canonically-sorted move
    assert decoded != _DECLARED  # ...which genuinely differs from request order here
    tera = action_to_order(22, battle)
    assert tera.order.id == _CANONICAL[0] and tera.terastallize


def test_round_trip_on_every_masked_action():
    battle = _live_battle()
    mask = action_mask(battle)
    legal = [a for a in range(GEN9_ACTION_SPACE_SIZE) if mask[a]]
    assert set(legal) >= {1, 6, 7, 8, 9, 22}  # bench switch, plain moves, tera variants
    for action in legal:
        assert order_to_action(action_to_order(action, battle), battle) == action


def test_mask_disables_the_canonical_slot_of_a_disabled_move():
    battle = _live_battle(disabled={"thunderbolt"})
    mask = action_mask(battle)
    slot = _CANONICAL.index("thunderbolt")
    assert not mask[6 + slot] and not mask[22 + slot]  # plain and tera variants both dead
    assert all(mask[6 + j] for j in range(4) if j != slot)


def test_lone_offset_available_move_claims_slot_six():
    battle = _live_battle(lone_move="struggle")
    mask = action_mask(battle)
    assert [a for a in range(6, 26) if mask[a]] == [6]  # special: no gimmick bits
    order = action_to_order(6, battle)
    assert order.order.id == "struggle"
    assert order_to_action(order, battle) == 6


def test_obs_move_blocks_align_with_action_slots():
    battle = _live_battle()
    obs = embed_battle(battle)
    ms, md = OBS_LAYOUT.moves_start, OBS_LAYOUT.move_dim
    for j in range(4):
        decoded = action_to_order(6 + j, battle).order
        assert obs[ms + j * md + md - 1] == vocab.move_id(decoded)


def test_illegal_action_falls_back_to_a_legal_order():
    battle = _live_battle(disabled={"thunderbolt"})
    slot = _CANONICAL.index("thunderbolt")
    order = action_to_order(6 + slot, battle)  # masked-out action must not raise or hang
    assert str(order) in [str(o) for o in battle.valid_orders]


def test_synthesized_mask_covers_canonical_slots():
    from lategame.features.action_space import synthesize_action_mask

    battle = Battle("battle-gen9rb-t2", "Alice", _LOGGER, gen=9)
    for line in (
        "|player|p1|Alice|1|",
        "|player|p2|Bob|2|",
        "|switch|p1a: Pikachu|Pikachu, L84, M|100/100",
        "|switch|p2a: Bulbasaur|Bulbasaur, L88, M|100/100",
        "|move|p1a: Pikachu|Volt Switch|p2a: Bulbasaur",
        "|move|p1a: Pikachu|Surf|p2a: Bulbasaur",
    ):
        battle.parse_message(line.split("|"))
    mask = synthesize_action_mask(battle, taken_action=7)
    assert mask.shape == (26,) and mask.dtype == bool
    assert mask[6] and mask[7] and not mask[8]  # two known moves -> two canonical slots

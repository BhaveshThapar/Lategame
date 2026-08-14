"""The doubles codec, and the two ways a factored action space is silently wrong.

Singles is 26 actions; doubles is 107 PER SLOT and a turn commits both. Modeling the joint action
would be 107^2 = 11,449 outputs, so the head is factored into two 107-way distributions. That
choice buys tractability and costs one thing: a factored mask cannot express a constraint that
COUPLES the slots, and there is exactly one that matters -- both slots switching to the same
benched Pokemon. Showdown rejects that order and answers with a default move rather than an error,
so an unchecked conflict is a lost turn that looks like a bad decision.

The other trap is inherited from Build 5: move slots must index CANONICAL move order (sorted by
move id), not poke-env's insertion order, or a slot means different moves at train and eval time.
"""

from __future__ import annotations

import numpy as np
from poke_env.battle import DoubleBattle, Pokemon
from poke_env.player import DefaultBattleOrder, ForfeitBattleOrder

from lategame.features.doubles_action_space import (
    GEN9_DOUBLES_ACTION_SPACE_SIZE,
    GEN9_DOUBLES_SLOT_ACTIONS,
    action_mask,
    decode_pokemon,
    joint_switch_conflict,
    order_to_action,
)


def test_the_factored_head_is_two_by_107_not_eleven_thousand():
    """The design claim, as an assertion: a factored head is 214 logits, a joint one 11,449."""
    assert GEN9_DOUBLES_SLOT_ACTIONS == 107
    assert GEN9_DOUBLES_ACTION_SPACE_SIZE == 214
    assert GEN9_DOUBLES_SLOT_ACTIONS**2 == 11449  # what factoring avoids


def test_singles_action_space_is_untouched():
    """Every existing checkpoint depends on the singles codec; adding doubles must not move it."""
    from lategame.features.action_space import GEN9_ACTION_SPACE_SIZE

    assert GEN9_ACTION_SPACE_SIZE == 26


class _FakeDoubleBattle(DoubleBattle):
    """Enough of a DoubleBattle for the codec; __init__ is skipped (it wants a websocket)."""

    def __init__(self, team, active, moves, switches, gen=9):
        self._t = {m.species: m for m in team}
        self._a, self._m, self._s, self._gen = active, moves, switches, gen
        # Read by DoublesEnv.get_action_mask_individual; __init__ is skipped so it is set here.
        self._wait = False
        self._teampreview = False
        self._finished = False
        self._player_role = 'p1'
        self._active_pokemon = {'p1a': active[0], 'p1b': active[1]}
        self._reviving = False
        # Read by the doubles encoder (features/doubles_encoder).
        self._opponent_team = {}
        self._teampreview_opponent_team = []
        self._weather = {}
        self._fields = {}
        self._side_conditions = {}
        self._opponent_side_conditions = {}
        self._turn = 3
        self._can_tera = [None, None]
        self._opponent_active_pokemon = {}
        self._maybe_trapped = [False, False]
        self._trapped = [False, False]
        self._force_switch = [False, False]
        self._can_mega_evolve = [False, False]
        self._can_z_move = [False, False]
        self._can_dynamax = [False, False]
        self._can_tera = [False, False]

    @property
    def team(self):
        return self._t

    @property
    def active_pokemon(self):
        return self._a

    @property
    def available_moves(self):
        return self._m

    @property
    def available_switches(self):
        return self._s

    @property
    def gen(self):
        return self._gen


def _battle():
    a0 = Pokemon(9, species="incineroar")
    for mid in ("knockoff", "fakeout", "partingshot", "flareblitz"):
        a0._add_move(mid)
    a1 = Pokemon(9, species="rillaboom")
    for mid in ("woodhammer", "grassyglide", "uturn", "fakeout"):
        a1._add_move(mid)
    bench = [Pokemon(9, species="amoonguss"), Pokemon(9, species="ironhands")]
    return _FakeDoubleBattle(
        team=[a0, a1, *bench],
        active=[a0, a1],
        moves=[list(a0.moves.values()), list(a1.moves.values())],
        switches=[bench, bench],
    )


def test_move_slots_follow_canonical_order_not_insertion_order():
    """Build 5's divergence, carried to doubles. Incineroar's moves were added knockoff-first;
    canonical order is alphabetical by move id, so slot 0 must be `fakeout`, not `knockoff`."""
    from lategame.features.doubles_action_space import _canonical_slot_moves

    b = _battle()
    assert [m.id for m in b.available_moves[0]][0] == "knockoff"  # insertion order
    assert [m.id for m in _canonical_slot_moves(b, 0)][0] == "fakeout"  # canonical


def test_the_mask_has_one_row_per_slot():
    mask = action_mask(_battle())
    assert mask.shape == (2, GEN9_DOUBLES_SLOT_ACTIONS)
    assert mask.dtype == bool
    assert mask.any(), "a battle with two healthy actives must offer something"


def test_both_slots_switching_to_one_pokemon_is_detected():
    """THE constraint a factored mask cannot represent. Showdown answers such an order with a
    default move rather than an error, so nothing downstream would notice."""
    b = _battle()
    assert joint_switch_conflict(np.array([1, 1]), b) is True   # both -> team[0]
    assert joint_switch_conflict(np.array([1, 2]), b) is False  # different bench mons
    assert joint_switch_conflict(np.array([7, 7]), b) is False  # both moves, not switches


def test_switch_actions_decode_to_the_named_pokemon():
    b = _battle()
    assert decode_pokemon(1, b).species == "incineroar"
    assert decode_pokemon(3, b).species == "amoonguss"
    assert decode_pokemon(7, b) is None  # a move, not a switch


def test_default_and_forfeit_use_the_poke_env_sentinels():
    b = _battle()
    assert list(order_to_action(DefaultBattleOrder(), b)) == [-2, -2]
    assert list(order_to_action(ForfeitBattleOrder(), b)) == [-1, -1]


def test_a_move_action_reindexes_symmetrically():
    """Round-trip through the reindexer: our canonical index -> poke-env's -> back must be
    identity, or a label written at collection time decodes to a different move at eval time."""
    from lategame.features.doubles_action_space import _MOVE_BASE, _reindex_move_action

    b = _battle()
    for slot in (0, 1):
        for move_idx in range(4):
            for target in range(5):
                ours = _MOVE_BASE + move_idx * 5 + target
                pe = _reindex_move_action(ours, b, slot, to_canonical=False)
                back = _reindex_move_action(pe, b, slot, to_canonical=True)
                assert back == ours, (slot, move_idx, target, ours, pe, back)

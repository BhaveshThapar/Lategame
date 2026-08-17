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
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player import DefaultBattleOrder, ForfeitBattleOrder

from rotomai.features.doubles_action_space import (
    GEN9_DOUBLES_ACTION_SPACE_SIZE,
    GEN9_DOUBLES_SLOT_ACTIONS,
    action_mask,
    action_to_order_checked,
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
    from rotomai.features.action_space import GEN9_ACTION_SPACE_SIZE

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
        # Read by poke-env only on its ERROR path (`_action_to_order_individual` interpolates
        # both into the ValueError message), so they are needed to exercise a decode FAILURE.
        self._player_username = 'p1'
        self._battle_tag = 'battle-gen9vgc2025regi-1'
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
    from rotomai.features.doubles_action_space import _canonical_slot_moves

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
    from rotomai.features.doubles_action_space import _MOVE_BASE, _reindex_move_action

    b = _battle()
    for slot in (0, 1):
        for move_idx in range(4):
            for target in range(5):
                ours = _MOVE_BASE + move_idx * 5 + target
                pe = _reindex_move_action(ours, b, slot, to_canonical=False)
                back = _reindex_move_action(pe, b, slot, to_canonical=True)
                assert back == ours, (slot, move_idx, target, ours, pe, back)


def test_a_half_default_becomes_a_pass_not_a_sentinel():
    """poke-env maps a DefaultBattleOrder HALF to -2 -- its whole-order sentinel -- but -2 has no
    meaning per slot, where "this slot does nothing" is action 0 (`pass`) in its own documented
    layout. 96% of collected VGC turns carry one such half (every partial replacement), and
    cross-entropy rejects them outright: `Target -2 is out of bounds`."""
    from rotomai.features.doubles_action_space import normalize_half_default

    assert list(normalize_half_default(np.array([12, -2]))) == [12, 0]
    assert list(normalize_half_default(np.array([-2, 5]))) == [0, 5]
    # A WHOLE default is a real "no decision" and must keep its sentinel for callers to drop.
    assert list(normalize_half_default(np.array([-2, -2]))) == [-2, -2]
    assert list(normalize_half_default(np.array([-1, -1]))) == [-1, -1]
    assert list(normalize_half_default(np.array([7, 9]))) == [7, 9]


def test_a_whole_default_order_keeps_its_sentinel():
    """Only a HALF-default is a pass; a whole DefaultBattleOrder still means 'no decision' and is
    dropped by callers rather than learned."""
    assert list(order_to_action(DefaultBattleOrder(), _battle())) == [-2, -2]


def test_the_mask_reads_poke_envs_per_index_flags_not_its_values():
    """`get_action_mask_individual` returns a length-107 0/1 mask indexed BY ACTION, not a list of
    legal indices. Iterating its values gave a 2-legal-action mask on a turn with 4 moves and 2
    switches -- which made 91% of collected slot-1 labels illegal under their own mask and drove
    BC's loss to 9.4e8, the model being trained toward a masked-out target."""
    b = _battle()
    ours = action_mask(b)
    for pos in (0, 1):
        raw = DoublesEnv.get_action_mask_individual(b, pos)
        assert len(raw) == GEN9_DOUBLES_SLOT_ACTIONS, "poke-env returns a per-index mask"
        assert int(ours[pos].sum()) == sum(raw), (
            f"slot {pos}: we marked {int(ours[pos].sum())} legal, poke-env says {sum(raw)}"
        )
    assert ours.sum() > 8, "two healthy actives with 4 moves each must offer well over 8 actions"


# --------------------------------------------------------------------------- #
# B6f: two poke-env INVARIANTS the factored sampler is built on.
#
# Both live in poke-env's `DoublesEnv.get_action_mask_individual`, not in our code, so a
# dependency bump could remove either without touching a line here. Pinned against poke-env
# directly, because the sequential sampler's correctness argument assumes them.
# --------------------------------------------------------------------------- #
def test_every_mask_row_always_offers_at_least_one_legal_action():
    """poke-env ends the builder with `actions = actions or [0]`, so `pass` is the floor. This is
    what makes a dead slot unreachable through the codec -- and what makes it safe for the
    sampler to delete slot 0's switch from slot 1's row."""
    b = _battle()
    mask = action_mask(b)
    assert mask.sum(axis=1).min() >= 1

    # ...including in the states that produce the degenerate rows: a partial replacement, where
    # the slot that was NOT asked may only pass.
    b._force_switch = [False, True]
    partial = action_mask(b)
    assert partial.sum(axis=1).min() >= 1
    assert partial[0].sum() == 1 and partial[0][0], "the unasked slot may only pass"


def test_pass_is_legal_in_the_one_state_where_restricting_could_empty_slot_one():
    """`all(force_switch)` with a single available switch is the only state where deleting slot
    0's switch from slot 1's row could leave it empty. poke-env already makes `pass` legal there
    (`switch_space + [0]`), so the restricted row is `{0}` and the resulting order is legal."""
    b = _battle()
    b._force_switch = [True, True]
    lone = [b.available_switches[0][0]]
    b._s = [lone, lone]
    mask = action_mask(b)
    assert mask[1][0], "pass must be legal, or the sequential draw has nothing left"
    assert mask[1].sum() == 2, "exactly the lone switch plus pass"


def test_the_only_mask_legal_pair_that_cannot_execute_is_the_joint_switch_conflict():
    """`action_to_order` falls back to a random legal order on a decode failure, SILENTLY. For an
    on-policy update that silence is a correctness hole -- the recorded action is then not the one
    played, so log pi_old describes something that never happened -- and the legality check cannot
    catch it, because the action WAS legal under its own mask.

    Swept over every mask-legal pair, the failures are EXACTLY the joint-switch conflicts. That is
    the design claim stated as a measurement: the factored mask expresses all legality except the
    one constraint coupling the slots, which is why `sample_factored_action` deletes slot 0's
    switch from slot 1's row -- and why, once it does, `invalid_frac` becoming non-zero is
    evidence of a NEW unmodelled constraint rather than a known tolerable loss.
    """
    b = _battle()
    mask = action_mask(b)
    checked = 0
    for a0 in np.where(mask[0])[0]:
        for a1 in np.where(mask[1])[0]:
            pair = np.array([a0, a1])
            _, executed = action_to_order_checked(pair, b)
            assert executed is not joint_switch_conflict(pair, b), (
                f"pair {pair.tolist()}: executed={executed} but "
                f"conflict={joint_switch_conflict(pair, b)}"
            )
            checked += 1
    assert checked > 20, "the sweep must actually cover the legal set"


def test_action_to_order_behaviour_is_unchanged_by_the_split():
    """The non-strict public entry point must still never raise -- every BC/AWR/eval caller
    depends on a mis-predicting policy not hanging a battle."""
    from rotomai.features.doubles_action_space import action_to_order

    b = _battle()
    assert action_to_order(np.array([3, 4]), b) is not None  # decodes
    assert action_to_order(np.array([3, 3]), b) is not None  # falls back rather than raising

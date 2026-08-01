"""Singles action codec: integer action <-> poke-env ``BattleOrder``, canonical move order.

The learned policy emits a discrete action; the environment needs a ``BattleOrder``. The
layout is poke-env 0.15's ``SinglesEnv`` convention (``get_action_space_size(9) == 26``)
with ONE deliberate divergence (Build 5): move slots index the active mon's known moves in
**canonical order** -- sorted by move id, first four -- instead of poke-env's
``list(active.moves.values())[:4]`` dict-insertion order. Insertion order differs between a
log-reconstructed battle (reveal order + sorted backfill) and a live battle (the
``|request|``'s declaration order), so positional slots meant different moves at train vs
eval time; canonical order is a pure function of the known-move *set*, identical on both
sides by construction. Switch slots keep poke-env's ``battle.team.values()`` order::

    0-5   switch to battle.team.values()[i]
    6-9   move i (order = canonical_moves(active): sorted by move id, first 4)
    10-13 move i + mega
    14-17 move i + z-move
    18-21 move i + dynamax
    22-25 move i + terastallize

The converters below mirror ``SinglesEnv``'s structure (fallbacks included) with
``canonical_moves`` swapped in. One preserved poke-env convention: when the request offers
exactly one available move that is NOT among the known four (Struggle, a locked/copied
move), it is masked, labelled, and decoded at slot 6 even though the slot-6 obs block
describes canonical move 0 -- a pre-existing misalignment kept for these rare forced states.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from poke_env.battle import AbstractBattle, Battle, Move, Pokemon
from poke_env.battle.move import SPECIAL_MOVES
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player import (
    BattleOrder,
    DefaultBattleOrder,
    ForfeitBattleOrder,
    Player,
    SingleBattleOrder,
)

# Gen-9 singles: 6 switches + 4 moves * (4 gimmicks + 1). Fixed for our format.
GEN9_ACTION_SPACE_SIZE: int = SinglesEnv.get_action_space_size(9)


def action_space_size(battle: AbstractBattle) -> int:
    """Action-space size for ``battle``'s generation."""
    return SinglesEnv.get_action_space_size(battle.gen)


def canonical_moves(mon: Pokemon) -> list[Move]:
    """``mon``'s known moves in canonical slot order: sorted by move id, first four.

    Move ids are unique within a moveset, so the sort is a total order over the known-move
    *set*, independent of dict insertion order (log reveal + backfill offline, request
    declaration live). Struggle/recharge and z/max moves never enter ``mon.moves``
    (``Move.should_be_stored``), so forced specials cannot perturb the slots.
    """
    return sorted(mon.moves.values(), key=lambda m: m.id)[:4]


def _slot_moves(battle: Battle) -> list[Move]:
    """The move list slots 6-9 index: canonical, or poke-env's lone off-set fallback."""
    active = battle.active_pokemon
    assert active is not None
    moves = canonical_moves(active)
    avail = battle.available_moves
    if len(avail) == 1 and avail[0].id not in [m.id for m in moves]:
        return list(avail)
    return moves


def _index_order(order: SingleBattleOrder, battle: Battle) -> int:
    """Positional action for a switch/move order; ``ValueError`` if un-indexable."""
    assert not isinstance(order.order, str)
    if isinstance(order.order, Pokemon):
        return [p.base_species for p in battle.team.values()].index(order.order.base_species)
    active = battle.active_pokemon
    if active is None:
        raise ValueError("move order with no active Pokemon")
    slot = [m.id for m in _slot_moves(battle)].index(order.order.id)
    if order.mega:
        gimmick = 1
    elif order.z_move:
        gimmick = 2
    elif order.dynamax:
        gimmick = 3
    elif order.terastallize:
        gimmick = 4
    else:
        gimmick = 0
    return 6 + slot + 4 * gimmick


# Singles battles are always the concrete `Battle`, whose request-derived members
# (available_moves, valid_orders, trapped, ...) the codec reads; the agent code passes the
# base `AbstractBattle`, so the narrowing is localized here.
def order_to_action(order: BattleOrder, battle: AbstractBattle) -> int:
    """Label a demonstrator's ``order`` with its integer action.

    Strict: raises ``ValueError`` if the order is not among the battle's legal orders, so
    callers (request-backed collection/resim) can drop ambiguous turns rather than record a
    corrupt label. ``DefaultBattleOrder`` -> -2, ``ForfeitBattleOrder`` -> -1 (poke-env
    convention; both are out of the learnable range and dropped by callers' guards).
    """
    b = cast(Battle, battle)
    if isinstance(order, DefaultBattleOrder):
        return -2
    if isinstance(order, ForfeitBattleOrder):
        return -1
    assert isinstance(order, SingleBattleOrder) and not isinstance(order.order, str)
    if str(order) not in [str(o) for o in b.valid_orders]:
        raise ValueError(f"order {order} not in valid orders")
    return _index_order(order, b)


def action_to_order(action: int, battle: AbstractBattle) -> BattleOrder:
    """Convert an integer ``action`` back into a ``BattleOrder`` for ``battle``.

    Non-strict: an out-of-context action falls back to a random legal move rather than
    raising, so a mis-predicting policy never hangs a battle.
    """
    b = cast(Battle, battle)
    try:
        if action == -2:
            return DefaultBattleOrder()
        if action == -1:
            return ForfeitBattleOrder()
        if action < 6:
            order: BattleOrder = Player.create_order(list(b.team.values())[action])
        else:
            if b.active_pokemon is None:
                raise ValueError("action names a move but no own Pokemon is active")
            mvs = _slot_moves(b)
            if (action - 6) % 4 >= len(mvs):
                raise ValueError(f"move slot {(action - 6) % 4} out of bounds")
            order = Player.create_order(
                mvs[(action - 6) % 4],
                mega=10 <= action < 14,
                z_move=14 <= action < 18,
                dynamax=18 <= action < 22,
                terastallize=22 <= action < 26,
            )
        if str(order) not in [str(o) for o in b.valid_orders]:
            raise ValueError(f"converted order {order} not in valid orders")
        return order
    except (ValueError, IndexError):
        return Player.choose_random_singles_move(b)


def action_mask(battle: AbstractBattle) -> np.ndarray:
    """Boolean legality mask over the action space (``True`` == legal), request-derived.

    Mirrors ``SinglesEnv.get_action_mask`` with canonical move slots, including both of its
    lone-available-move fallback positions: a non-special lone move claims slot 6 *before*
    the gimmick spaces are derived (so they inherit it), a special one (Struggle/recharge)
    *after* (no gimmick variants of a forced special).
    """
    b = cast(Battle, battle)
    switch_space = [
        i
        for i, mon in enumerate(b.team.values())
        if not b.trapped and mon.base_species in [p.base_species for p in b.available_switches]
    ]
    if b._wait:
        actions = [0]
    elif b.active_pokemon is None:
        actions = switch_space
    else:
        known_moves = canonical_moves(b.active_pokemon)
        avail_move_ids = [m.id for m in b.available_moves]
        move_space = [6 + i for i, move in enumerate(known_moves) if move.id in avail_move_ids]
        if (
            not move_space
            and len(b.available_moves) == 1
            and b.available_moves[0].id not in SPECIAL_MOVES
        ):
            move_space = [6]
        available_z_ids = [m.id for m in b.active_pokemon.available_z_moves]
        mega_space = [i + 4 for i in move_space if b.can_mega_evolve]
        zmove_space = [
            i + 14
            for i, move in enumerate(known_moves)
            if b.can_z_move and move.id in avail_move_ids and move.id in available_z_ids
        ]
        dynamax_space = [i + 12 for i in move_space if b.can_dynamax]
        tera_space = [i + 16 for i in move_space if b.can_tera]
        if (
            not move_space
            and len(b.available_moves) == 1
            and b.available_moves[0].id in SPECIAL_MOVES
        ):
            move_space = [6]
        actions = (
            switch_space + move_space + mega_space + zmove_space + dynamax_space + tera_space
        )
    mask = np.zeros(action_space_size(battle), dtype=bool)
    mask[actions] = True
    return mask


# --------------------------------------------------------------------------- #
# Request-free siblings for offline replay ingestion (M6).
#
# Public Showdown replays carry no ``|request|`` JSON, so a battle reconstructed
# from spectator logs has empty ``available_moves``/``available_switches`` and the
# functions above (which read them) can't be used. These two recover the same
# labels from the *public* state instead -- switch from ``team.values()`` order,
# move from ``canonical_moves`` -- the exact orderings ``embed_battle`` and the
# live converters use, so ingested samples align with live slot semantics.
# --------------------------------------------------------------------------- #


def label_action(order: BattleOrder, battle: AbstractBattle) -> int:
    """Label ``order`` with its integer action from *public* battle state only.

    Skips the request-derived ``valid_orders`` legality guard (a reconstructed battle has
    no request). Returns a negative action when the order cannot be indexed (move beyond
    the canonical four, unknown switch target); callers treat negatives as undecodable and
    drop the turn.
    """
    b = cast(Battle, battle)
    if isinstance(order, DefaultBattleOrder):
        return -2
    if isinstance(order, ForfeitBattleOrder):
        return -1
    assert isinstance(order, SingleBattleOrder) and not isinstance(order.order, str)
    try:
        return _index_order(order, b)
    except ValueError:
        return -2


def synthesize_action_mask(battle: AbstractBattle, taken_action: int) -> np.ndarray:
    """Permissive legality mask for a replay turn (no ``|request|`` available).

    Marks every canonical move slot (-> 6-9) and every non-active, non-fainted team slot
    (``team.values()`` -> 0-5) legal, plus the actually-taken action. Over-approximates
    legality, which is harmless for the masked CE / AWR losses: they only require the
    *taken* action to stay unmasked.
    """
    mask = np.zeros(GEN9_ACTION_SPACE_SIZE, dtype=bool)
    for i, mon in enumerate(battle.team.values()):
        if i < 6 and not mon.active and not mon.fainted:
            mask[i] = True
    active = battle.active_pokemon
    if active is not None:
        for i in range(len(canonical_moves(active))):
            mask[6 + i] = True
    if 0 <= taken_action < GEN9_ACTION_SPACE_SIZE:
        mask[taken_action] = True
    return mask

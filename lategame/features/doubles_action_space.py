"""Doubles action codec: a FACTORED per-slot action pair <-> poke-env ``DoubleBattleOrder``.

The singles counterpart (``features/action_space``) wraps ``SinglesEnv``; this wraps
``DoublesEnv``, and the shape of the problem is the reason G4 asks for "≥3 formats ... without
rewriting the core":

* Singles is **26** actions (``SinglesEnv.get_action_space_size(9)``).
* Doubles is **107 per slot**, and a turn commits BOTH slots. Modeling the joint action directly
  would be 107² = **11,449** outputs, almost all of which are never legal together. The head is
  therefore **factored**: two independent distributions of 107, i.e. 214 logits, which is also the
  shape ``DoublesEnv`` already speaks (its converters take an array of two).

poke-env's per-slot layout, reproduced here because the codec below depends on it::

    -2       default          -1       forfeit          0        pass
    1-6      switch to team[i]
    7-106    move m (1-4) x target (-2,-1,0,1,2) x gimmick (none/mega/z/dyna/tera)

ONE DELIBERATE DIVERGENCE, INHERITED FROM SINGLES (Build 5). Move slots index the active mon's
known moves in **canonical order** -- sorted by move id, first four -- not poke-env's
``list(active.moves.values())[:4]`` dict-insertion order. Insertion order differs between a
log-reconstructed battle (reveal order + sorted backfill) and a live one (the ``|request|``'s
declaration order), so a positional slot would mean different moves at train and eval time. The
divergence is load-bearing and is carried here rather than re-derived.

The both-slots-cannot-switch-to-one-Pokemon constraint is enforced too: Showdown rejects such an
order and answers with a default move rather than an error, which is a silent lost turn -- the
same trap ``forward_driver.legalChoices`` and ``heuristic_agent.doubles_pick`` already guard.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from poke_env.battle import AbstractBattle, DoubleBattle, Move, Pokemon
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player import BattleOrder, DefaultBattleOrder, ForfeitBattleOrder

from lategame.features.action_space import canonical_moves

#: Actions available to ONE slot. 1 pass + 6 switches + 4 moves x 5 targets x 5 gimmicks.
GEN9_DOUBLES_SLOT_ACTIONS: int = DoublesEnv.get_action_space_size(9)

#: Logits a factored doubles head emits: one distribution per slot.
GEN9_DOUBLES_ACTION_SPACE_SIZE: int = 2 * GEN9_DOUBLES_SLOT_ACTIONS

#: Slot-layout constants, named rather than spelled as magic numbers at the call sites.
_PASS = 0
_SWITCH_BASE = 1
_MOVE_BASE = 7
_N_TARGETS = 5
_N_MOVES = 4


def slot_actions(battle: AbstractBattle) -> int:
    """Per-slot action count for ``battle``'s generation."""
    return DoublesEnv.get_action_space_size(battle.gen)


def _canonical_slot_moves(battle: DoubleBattle, pos: int) -> list[Move]:
    """The move list slots 7-106 index for ``pos``: canonical, with poke-env's lone fallback.

    Mirrors ``action_space._slot_moves``: when the request offers exactly one move that is not
    among the known four (Struggle, a locked or copied move), that move takes slot 0 -- the same
    rare-forced-state convention singles preserves.
    """
    active = battle.active_pokemon[pos] if pos < len(battle.active_pokemon) else None
    if active is None:
        return []
    moves = canonical_moves(active)
    avail = list(battle.available_moves[pos]) if pos < len(battle.available_moves) else []
    if len(avail) == 1 and avail[0].id not in [m.id for m in moves]:
        return avail
    return moves


def _reindex_move_action(action: int, battle: DoubleBattle, pos: int, to_canonical: bool) -> int:
    """Translate a move action between poke-env's move order and ours.

    poke-env indexes move slots by ``available_moves[pos]``; we index by ``canonical_moves``.
    Only the move component moves -- target and gimmick are untouched.
    """
    if action < _MOVE_BASE:
        return action
    offset = action - _MOVE_BASE
    gimmick, rest = divmod(offset, _N_MOVES * _N_TARGETS)
    move_idx, target = divmod(rest, _N_TARGETS)

    pe_moves = list(battle.available_moves[pos]) if pos < len(battle.available_moves) else []
    our_moves = _canonical_slot_moves(battle, pos)
    src, dst = (pe_moves, our_moves) if to_canonical else (our_moves, pe_moves)
    if move_idx >= len(src):
        return action
    try:
        move_idx = [m.id for m in dst].index(src[move_idx].id)
    except ValueError:
        return action
    return _MOVE_BASE + gimmick * _N_MOVES * _N_TARGETS + move_idx * _N_TARGETS + target


def order_to_action(order: BattleOrder, battle: AbstractBattle) -> np.ndarray:
    """Label a demonstrator's doubles ``order`` with its per-slot action pair.

    Strict, like the singles codec: raises ``ValueError`` on an order that is not legal here, so
    request-backed collection drops an ambiguous turn rather than recording a corrupt label.
    ``DefaultBattleOrder`` -> ``[-2, -2]``, ``ForfeitBattleOrder`` -> ``[-1, -1]``.
    """
    b = cast(DoubleBattle, battle)
    if isinstance(order, DefaultBattleOrder):
        return np.array([-2, -2], dtype=np.int64)
    if isinstance(order, ForfeitBattleOrder):
        return np.array([-1, -1], dtype=np.int64)
    pe = DoublesEnv.order_to_action(order, b, strict=True)
    return np.array(
        [_reindex_move_action(int(pe[i]), b, i, to_canonical=True) for i in (0, 1)],
        dtype=np.int64,
    )


def action_to_order(action: np.ndarray, battle: AbstractBattle) -> BattleOrder:
    """Convert a per-slot action pair back into a ``DoubleBattleOrder``.

    Non-strict, like the singles codec: an out-of-context action falls back to a random legal
    doubles order rather than raising, so a mis-predicting policy never hangs a battle.
    """
    from poke_env.player import Player

    b = cast(DoubleBattle, battle)
    try:
        pe = np.array(
            [_reindex_move_action(int(action[i]), b, i, to_canonical=False) for i in (0, 1)],
            dtype=np.int64,
        )
        return DoublesEnv.action_to_order(pe, b, strict=True)
    except (ValueError, IndexError, AssertionError):
        return Player.choose_random_doubles_move(b)


def action_mask(battle: AbstractBattle) -> np.ndarray:
    """Boolean legality mask, shape ``(2, slot_actions)`` -- one row per slot.

    Built from ``DoublesEnv.get_action_mask_individual`` and then re-indexed into canonical move
    order, so the mask and the codec agree by construction rather than by parallel maintenance.
    """
    b = cast(DoubleBattle, battle)
    n = slot_actions(b)
    mask = np.zeros((2, n), dtype=bool)
    for pos in (0, 1):
        for a in DoublesEnv.get_action_mask_individual(b, pos):
            if 0 <= a < n:
                mask[pos, _reindex_move_action(int(a), b, pos, to_canonical=True)] = True
    return mask


def joint_switch_conflict(action: np.ndarray, battle: AbstractBattle) -> bool:
    """Whether both slots switch to the SAME benched Pokemon -- an order Showdown rejects.

    Not expressible in a factored mask (it couples the two slots), so it is a separate check the
    decoder and any sampler must apply. Showdown answers such an order with a default move rather
    than an error, so an unchecked conflict is a silently lost turn.
    """
    a0, a1 = int(action[0]), int(action[1])
    switch = range(_SWITCH_BASE, _SWITCH_BASE + 6)
    return a0 in switch and a1 in switch and a0 == a1


def decode_pokemon(action: int, battle: AbstractBattle) -> Pokemon | None:
    """The Pokemon a switch action names, or ``None`` if the action is not a switch."""
    b = cast(DoubleBattle, battle)
    if not _SWITCH_BASE <= action < _SWITCH_BASE + 6:
        return None
    team = list(b.team.values())
    idx = action - _SWITCH_BASE
    return team[idx] if idx < len(team) else None


__all__ = [
    "GEN9_DOUBLES_ACTION_SPACE_SIZE",
    "GEN9_DOUBLES_SLOT_ACTIONS",
    "action_mask",
    "action_to_order",
    "decode_pokemon",
    "joint_switch_conflict",
    "order_to_action",
    "slot_actions",
]

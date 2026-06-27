"""Singles action codec: integer action <-> poke-env ``BattleOrder``.

The learned policy emits a discrete action; the environment needs a
``BattleOrder``. Rather than invent a mapping, we reuse poke-env 0.15's own
``SinglesEnv`` convention so our action space is identical to the one the
library (and any future RL env) uses. ``SinglesEnv``'s converters are
``@staticmethod``s, so we delegate to them directly -- no env instance, no risk
of the mapping drifting out of sync.

Gen-9 singles layout (``get_action_space_size(9) == 26``)::

    0-5   switch to battle.team.values()[i]
    6-9   move i (order = list(active.moves.values())[:4])
    10-13 move i + mega
    14-17 move i + z-move
    18-21 move i + dynamax
    22-25 move i + terastallize

Our M2 demonstrators only ever pick plain moves/switches, so collected labels
land in 0-9; the higher gimmick slots exist for parity but stay unused until a
demonstrator (or the policy) uses them.
"""

from __future__ import annotations

import numpy as np
from poke_env.battle import AbstractBattle
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player import BattleOrder

# Gen-9 singles: 6 switches + 4 moves * (4 gimmicks + 1). Fixed for our format.
GEN9_ACTION_SPACE_SIZE: int = SinglesEnv.get_action_space_size(9)


def action_space_size(battle: AbstractBattle) -> int:
    """Action-space size for ``battle``'s generation."""
    return SinglesEnv.get_action_space_size(battle.gen)


# poke-env's SinglesEnv codec is typed for the concrete `Battle`; singles
# battles always are one, so we accept the base `AbstractBattle` (used throughout
# the agent code) and localize the type narrowing to these three delegations.
def order_to_action(order: BattleOrder, battle: AbstractBattle) -> int:
    """Label a demonstrator's ``order`` with its integer action.

    Strict: raises ``ValueError`` if the order is not among the battle's legal
    orders, so callers (data collection) can drop ambiguous turns rather than
    record a corrupt label.
    """
    return int(SinglesEnv.order_to_action(order, battle, strict=True))  # type: ignore[arg-type]


def action_to_order(action: int, battle: AbstractBattle) -> BattleOrder:
    """Convert an integer ``action`` back into a ``BattleOrder`` for ``battle``.

    Non-strict: an out-of-context action falls back to a random legal move
    rather than raising, so a mis-predicting policy never hangs a battle.
    """
    return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)  # type: ignore[arg-type]


def action_mask(battle: AbstractBattle) -> np.ndarray:
    """Boolean legality mask over the action space (``True`` == legal)."""
    return np.asarray(SinglesEnv.get_action_mask(battle), dtype=bool)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Request-free siblings for offline replay ingestion (M6).
#
# Public Showdown replays carry no ``|request|`` JSON, so a battle reconstructed
# from spectator logs has empty ``available_moves``/``available_switches`` and the
# functions above (which read them) can't be used. These two recover the same
# labels positionally from the *public* state instead -- the only state a replay
# gives us -- so ingested samples align with the live encoder's move/switch order.
# --------------------------------------------------------------------------- #


def label_action(order: BattleOrder, battle: AbstractBattle) -> int:
    """Label ``order`` with its integer action from *public* battle state only.

    Uses poke-env's codec with ``fake=True``, which skips the request-derived
    ``valid_orders`` legality guard and computes the index purely positionally --
    switch from ``battle.team.values()`` order, move from ``active.moves`` order,
    the same orderings ``embed_battle`` uses. ``strict=False`` so an
    un-indexable order returns a default rather than raising; callers should
    treat a default (negative) result as undecodable and drop the turn.
    """
    return int(SinglesEnv.order_to_action(order, battle, fake=True, strict=False))  # type: ignore[arg-type]


def synthesize_action_mask(battle: AbstractBattle, taken_action: int) -> np.ndarray:
    """Permissive legality mask for a replay turn (no ``|request|`` available).

    Marks every revealed move slot (``active.moves[:4]`` -> 6-9) and every
    non-active, non-fainted team slot (``team.values()`` -> 0-5) legal, plus the
    actually-taken action. Over-approximates legality, which is harmless for the
    masked CE / AWR losses: they only require the *taken* action to stay unmasked.
    """
    mask = np.zeros(GEN9_ACTION_SPACE_SIZE, dtype=bool)
    for i, mon in enumerate(battle.team.values()):
        if i < 6 and not mon.active and not mon.fainted:
            mask[i] = True
    active = battle.active_pokemon
    if active is not None:
        for i in range(min(len(active.moves), 4)):
            mask[6 + i] = True
    if 0 <= taken_action < GEN9_ACTION_SPACE_SIZE:
        mask[taken_action] = True
    return mask

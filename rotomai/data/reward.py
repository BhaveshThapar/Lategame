"""Shaped state-value for offline RL rewards (plan.md, M3).

The reward for a turn is the change in a scalar *state value* between consecutive
decision points. ``state_value`` is a pure replica of poke-env's
``PokeEnv.reward_computing_helper`` state-value formula (the ``current_value``
accumulator), minus the internal buffer -- so we can evaluate it on any battle
object without instantiating an env. Diffing successive values reproduces the
exact reward poke-env would emit, and a discounted sum of those diffs equals
``state_value(terminal) - state_value(start)``.

All shaping weights live in ``RewardWeights``; setting the shaping terms to 0 and
keeping ``victory_value`` recovers the sparse win/loss objective.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from poke_env.battle import AbstractBattle


@dataclass(frozen=True)
class RewardWeights:
    """Per-factor weights for the shaped state value (own = negative, opp = positive)."""

    hp_value: float = 0.3
    fainted_value: float = 1.0
    status_value: float = 0.1
    victory_value: float = 1.0
    number_of_pokemons: int = 6


def digest_state_value(
    digest: Mapping[str, Any],
    weights: RewardWeights,
    *,
    winner: str | None = None,
) -> float:
    """``state_value`` computed from a forward-driver DIGEST instead of a poke-env battle.

    Deliberately adjacent to ``state_value`` and a line-for-line mirror of it, because the two
    must agree -- ``tests/test_reward.py`` pins that they do. Kept as a second implementation
    rather than a refactor because the inputs are genuinely different objects: one walks poke-env
    ``Pokemon``, the other a JSON digest the node driver already produced.

    **Why it exists.** Scoring a shaped search leaf through poke-env means deepcopying the live
    battle and replaying the fork's delta line by line -- ~170 ms per node, against ~1 ms for the
    node RPC that produced the digest in the first place. That single cost is what made a depth-2
    VGC search ~190 s/battle and put the M2 ceiling probe out of reach.

    The digest is always in the DRIVER frame (``p1`` is us -- see ``determinize.battle_to_spec``),
    and its HP is rounded to two decimals, so agreement with ``state_value`` is exact only to
    within that rounding.
    """
    value = 0.0
    ours = digest.get("p1", {})
    theirs = digest.get("p2", {})

    for mon in ours.values():
        value += float(mon.get("hp", 0.0)) * weights.hp_value
        if mon.get("fainted"):
            value -= weights.fainted_value
        elif mon.get("status"):
            value -= weights.status_value
    value += (weights.number_of_pokemons - len(ours)) * weights.hp_value

    for mon in theirs.values():
        value -= float(mon.get("hp", 0.0)) * weights.hp_value
        if mon.get("fainted"):
            value += weights.fainted_value
        elif mon.get("status"):
            value += weights.status_value
    value -= (weights.number_of_pokemons - len(theirs)) * weights.hp_value

    if winner == "p1":
        value += weights.victory_value
    elif winner == "p2":
        value -= weights.victory_value
    return value


def state_value(battle: AbstractBattle, weights: RewardWeights) -> float:
    """Scalar value of ``battle`` from our POV (higher = better for us).

    Mirrors poke-env ``reward_computing_helper``: our team's HP/faints/status count
    negative, the opponent's positive, unrevealed mons are credited full HP, and the
    terminal win/loss adds ``+/- victory_value`` (only once ``battle.won/lost`` is set).
    """
    value = 0.0

    for mon in battle.team.values():
        value += mon.current_hp_fraction * weights.hp_value
        if mon.fainted:
            value -= weights.fainted_value
        elif mon.status is not None:
            value -= weights.status_value
    value += (weights.number_of_pokemons - len(battle.team)) * weights.hp_value

    for mon in battle.opponent_team.values():
        value -= mon.current_hp_fraction * weights.hp_value
        if mon.fainted:
            value += weights.fainted_value
        elif mon.status is not None:
            value += weights.status_value
    value -= (weights.number_of_pokemons - len(battle.opponent_team)) * weights.hp_value

    if battle.won:
        value += weights.victory_value
    elif battle.lost:
        value -= weights.victory_value

    return value

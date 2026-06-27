"""R-CALC seed: lightweight expected-damage / move-value estimation.

This is a *seed*, not the full damage formula. A move is scored by an
expected-damage proxy:

    base_power x STAB x type_effectiveness x accuracy

It deliberately does NOT model damage rolls, crits, weather/item/ability
modifiers, exact stats, or OHKO/2HKO probabilities. The full R-CALC engine
(a port of foul-play's Rust `poke-engine`, with damage ranges and KO odds)
is a later milestone -- see plan.md, requirement R-CALC.

The pure helpers (`stab_multiplier`, `normalize_accuracy`, `expected_power`)
are dependency-free so they can be unit-tested without poke-env objects.
"""

from __future__ import annotations

from collections.abc import Iterable

from poke_env.battle import Move, MoveCategory, Pokemon, PokemonType

STAB_MULTIPLIER = 1.5


def stab_multiplier(
    move_type: PokemonType | None,
    attacker_types: Iterable[PokemonType | None],
) -> float:
    """1.5x Same-Type Attack Bonus when the move shares a type with the attacker."""
    if move_type is None:
        return 1.0
    return STAB_MULTIPLIER if move_type in set(attacker_types) else 1.0


def normalize_accuracy(accuracy: bool | float | None) -> float:
    """poke-env reports never-miss moves as ``True``; map everything to [0, 1]."""
    if accuracy is True or accuracy is None:
        return 1.0
    return float(accuracy)


def expected_power(
    base_power: float, stab: float, type_multiplier: float, accuracy: float
) -> float:
    """The core expected-damage proxy. Pure and side-effect free."""
    return base_power * stab * type_multiplier * accuracy


def score_move(move: Move, attacker: Pokemon, defender: Pokemon | None) -> float:
    """Expected-value score for using ``move`` from ``attacker`` into ``defender``.

    Status (non-damaging) moves score 0 here -- the policy treats them as a
    fallback rather than valuing them through this damage proxy.
    """
    base_power = move.base_power or 0
    if move.category == MoveCategory.STATUS or base_power == 0:
        return 0.0

    type_multiplier = defender.damage_multiplier(move) if defender is not None else 1.0
    stab = stab_multiplier(move.type, attacker.types)
    accuracy = normalize_accuracy(move.accuracy)
    return expected_power(base_power, stab, type_multiplier, accuracy)

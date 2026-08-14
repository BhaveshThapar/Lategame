"""Doubles observation encoder: the singles blocks, emitted for TWO active slots per side.

The counterpart of ``features/encoder`` for doubles, and deliberately a thin one -- every per-mon
and per-move block builder is imported from there rather than re-derived, because G4's claim is
that a third format plugs in *without rewriting the core*. What genuinely differs is only:

* **Move blocks are per active slot.** Singles emits ``n_moves`` blocks describing "the" active's
  moves; doubles emits ``n_active * n_moves``, so a policy can score each slot's options.
* **The global block is per slot where the game is.** ``force_switch``, ``maybe_trapped`` and
  ``first_turn`` are per-slot lists on a ``DoubleBattle``, and which of the two FOES is still on
  the field is the targeting context a factored per-slot head needs to choose a target at all.

``OBS_VERSION_DOUBLES`` is separate from the singles ``OBS_VERSION`` on purpose. Datasets and
checkpoints are fingerprinted by (version, dim) -- ``data.collect`` refuses a shard whose
fingerprint does not match -- so a doubles shard can never be silently loaded by a singles run,
and every existing singles checkpoint keeps its exact layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from poke_env.battle import AbstractBattle, Pokemon

from lategame.features import vocab
from lategame.features.action_space import canonical_moves
from lategame.features.encoder import (
    _FIELDS,
    _MOVE_DIM,
    _N_MOVES,
    _POKEMON_DIM,
    _POKEMON_ID_CHANNELS,
    _SIDE_CONDITIONS,
    _TEAM_SIZE,
    _TURN_SCALE,
    _WEATHERS,
    MOVE_ID_FIELDS,
    ObsLayout,
    _encode_move,
    _encode_pokemon,
    _opponent_mons,
    _padded,
    _presence,
)

#: Active Pokemon per side.
N_ACTIVE = 2

#: Per-slot and shared scalars. Singles has 5; doubles needs them per slot plus the targeting
#: context (which foe slots are still on the field), which singles has no notion of.
_GLOBAL_SCALARS_DOUBLES = 12

_GLOBAL_DIM_DOUBLES = (
    len(_WEATHERS) + len(_FIELDS) + 2 * len(_SIDE_CONDITIONS) + _GLOBAL_SCALARS_DOUBLES
)

OBS_DIM_DOUBLES = (
    _TEAM_SIZE * 2 * _POKEMON_DIM + N_ACTIVE * _N_MOVES * _MOVE_DIM + _GLOBAL_DIM_DOUBLES
)

OBS_VERSION_DOUBLES = f"d1-{vocab.vocab_version()}"

OBS_LAYOUT_DOUBLES = ObsLayout(
    team_size=_TEAM_SIZE,
    n_moves=_N_MOVES,
    pokemon_dim=_POKEMON_DIM,
    move_dim=_MOVE_DIM,
    global_dim=_GLOBAL_DIM_DOUBLES,
    pokemon_id_channels=_POKEMON_ID_CHANNELS,
    move_id_channels=len(MOVE_ID_FIELDS),
    n_active=N_ACTIVE,
)


def _slot_mon(seq: Any, i: int) -> Pokemon | None:
    """``seq[i]`` when ``seq`` is a per-slot list, else ``seq`` itself (defensive on POV shape)."""
    if isinstance(seq, list):
        return seq[i] if i < len(seq) else None
    return seq


def _slot_flag(seq: Any, i: int) -> bool:
    """A per-slot boolean, tolerating either a list (doubles) or a scalar (defensive)."""
    if isinstance(seq, list):
        return bool(seq[i]) if i < len(seq) else False
    return bool(seq)


def _encode_global_doubles(battle: Any) -> np.ndarray:
    weather = _presence(battle.weather, _WEATHERS)
    fields = _presence(battle.fields, _FIELDS)
    own_side = _presence(battle.side_conditions, _SIDE_CONDITIONS)
    opp_side = _presence(battle.opponent_side_conditions, _SIDE_CONDITIONS)

    ours = [_slot_mon(battle.active_pokemon, i) for i in range(N_ACTIVE)]
    foes = [_slot_mon(battle.opponent_active_pokemon, i) for i in range(N_ACTIVE)]
    force = [_slot_flag(battle.force_switch, i) for i in range(N_ACTIVE)]
    trapped = [_slot_flag(battle.maybe_trapped, i) for i in range(N_ACTIVE)]

    flags = np.array(
        [
            battle.turn / _TURN_SCALE,
            float(force[0]),
            float(force[1]),
            float(bool(any(battle.can_tera) if isinstance(battle.can_tera, list)
                       else battle.can_tera)),
            float(trapped[0]),
            float(trapped[1]),
            # Build 9 Gate B's recency signal, per slot: "just switched, hasn't acted".
            float(getattr(ours[0], "first_turn", 0.0) or 0.0),
            float(getattr(ours[1], "first_turn", 0.0) or 0.0),
            # Which of OUR slots is occupied, and which of the FOE's. The second pair is the
            # targeting context: a factored per-slot head picks a target slot, and a target that
            # is empty is not a legal choice -- without this the policy cannot tell.
            float(ours[0] is not None),
            float(ours[1] is not None),
            float(foes[0] is not None),
            float(foes[1] is not None),
        ],
        dtype=np.float32,
    )
    return np.concatenate([weather, fields, own_side, opp_side, flags])


def embed_doubles_battle(battle: AbstractBattle) -> np.ndarray:
    """Encode a doubles POV into a fixed-size ``float32`` vector of ``OBS_DIM_DOUBLES``."""
    b: Any = battle  # per-slot shape is handled by _slot_mon / _slot_flag
    team = _padded(b.team.values(), _TEAM_SIZE)
    opp_team = _padded(_opponent_mons(b), _TEAM_SIZE)

    blocks = [_encode_pokemon(m) for m in team]
    blocks += [_encode_pokemon(m) for m in opp_team]

    # One move block group per active slot, in slot order. An empty slot contributes zeros, which
    # is the same representation a missing move already has -- so "no mon here" and "no move here"
    # are consistent rather than two different kinds of absence.
    for i in range(N_ACTIVE):
        active = _slot_mon(b.active_pokemon, i)
        # A doubles move's value depends on WHICH foe it hits; the encoder scores against the
        # opposing mon in the same slot, and the policy reads both foes from the mon blocks.
        defender = _slot_mon(b.opponent_active_pokemon, i)
        moves = _padded(canonical_moves(active) if active is not None else [], _N_MOVES)
        blocks += [_encode_move(m, active, defender) for m in moves]

    blocks.append(_encode_global_doubles(b))

    obs = np.concatenate(blocks)
    assert obs.shape == (OBS_DIM_DOUBLES,), (
        f"doubles encoder produced {obs.shape}, expected {(OBS_DIM_DOUBLES,)}"
    )
    return obs


__all__ = [
    "N_ACTIVE",
    "OBS_DIM_DOUBLES",
    "OBS_LAYOUT_DOUBLES",
    "OBS_VERSION_DOUBLES",
    "embed_doubles_battle",
]

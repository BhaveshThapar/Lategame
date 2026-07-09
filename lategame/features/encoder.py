"""Battle -> fixed-size feature vector (R-ENCODE seed, simple-first).

``embed_battle`` turns a poke-env battle (from our POV) into a flat ``float32``
vector the BC policy consumes. This is the *simple-first* encoder agreed for M2:
one-hots + numerics only, **no species/move/item embedding tables** -- those are
the transformer-era upgrade (plan.md, R-ENCODE). The expected-damage signal from
the R-CALC seed (``engine.damage.score_move``) is folded in per move so the model
gets the damage proxy for free.

Layout (all dims derived from poke-env enums, so they can't silently drift):

* 12 Pokemon blocks: 6 ego (``battle.team``) + 6 opponent (``battle.opponent_team``),
  zero-padded for empty/unrevealed slots.
* 4 move blocks for the ego active Pokemon, in canonical order (``canonical_moves``:
  sorted by move id, first four) -- the same order the action codec uses, so move
  features align with move action logits regardless of the ``moves`` dict's insertion
  order (log reveal offline vs request declaration live).
* 1 global block: weather / field / both sides' hazards+screens / turn / flags.

``OBS_VERSION`` + ``OBS_DIM`` are stamped into datasets and checkpoints; loaders
assert they match the live encoder so a stale encoding can never be used silently.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from poke_env.battle import (
    AbstractBattle,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)
from poke_env.battle.side_condition import STACKABLE_CONDITIONS

from lategame.engine.damage import normalize_accuracy, score_move
from lategame.features import vocab
from lategame.features.action_space import canonical_moves

# Folds the vocab content hash into the version, so the existing obs-version guard on
# shards/checkpoints also rejects an obs encoded against a different identity vocab.
# v3 = Build-5 canonical move-slot order: slot semantics changed, so every v2-era shard
# and checkpoint must be rejected rather than silently mix orderings.
# v4 = Build-9 Gate A (superseded): pp_fraction ablated to a constant -> BC RED (pp is
# load-bearing for imitation), so pp is kept.
# v5 = Build-9 Gate B: adds an explicit own-active first_turn recency channel to the global
# block (pp retained) -- the global block widens, so v4-era shards/checkpoints must be rejected.
OBS_VERSION = f"v5-{vocab.vocab_version()}"

# Stable index maps. Enum iteration order is fixed (aliases excluded), so these
# define a deterministic, versioned feature layout.
_TYPES: list[PokemonType] = list(PokemonType)
_STATUSES: list[Status] = list(Status)
_WEATHERS: list[Weather] = list(Weather)
_FIELDS: list[Field] = list(Field)
_SIDE_CONDITIONS: list[SideCondition] = list(SideCondition)
_CATEGORIES: list[MoveCategory] = list(MoveCategory)

_TYPE_INDEX = {t: i for i, t in enumerate(_TYPES)}
_STATUS_INDEX = {s: i for i, s in enumerate(_STATUSES)}
_CATEGORY_INDEX = {c: i for i, c in enumerate(_CATEGORIES)}

_BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")

_TEAM_SIZE = 6
_N_MOVES = 4

# Normalisation constants (rough scales -- keeps features ~O(1)).
_BP_SCALE = 150.0
_PRIORITY_SCALE = 7.0
_STAT_SCALE = 255.0
_BOOST_SCALE = 6.0
_TURN_SCALE = 100.0
_SCORE_SCALE = 300.0

# Scalar (non-one-hot) feature counts per block, named so the dims can't drift
# from the arrays built below.
_POKEMON_SCALARS = 4  # present, active, fainted, hp_fraction
_MOVE_SCALARS = 6  # present, base_power, accuracy, priority, pp_fraction, score
_GLOBAL_SCALARS = 5  # turn, force_switch, can_tera, maybe_trapped, own_first_turn

# Identity ID channels (R-ENCODE): trailing integer channels per entity block, each a
# ``vocab`` index (0 = none/unknown) the model looks up in a learned embedding table.
# Kept at the END of every block so the "present" flag stays at index 0 and the numeric
# features stay contiguous -- the transformer splits ``[numeric | ids]`` by these counts.
POKEMON_ID_FIELDS = ("species", "items", "abilities")
MOVE_ID_FIELDS = ("moves",)
_POKEMON_ID_CHANNELS = len(POKEMON_ID_FIELDS)
_MOVE_ID_CHANNELS = len(MOVE_ID_FIELDS)

_POKEMON_DIM = (
    _POKEMON_SCALARS
    + len(_TYPES)
    + len(_STATUSES)
    + len(_BOOST_KEYS)
    + len(_STAT_KEYS)
    + _POKEMON_ID_CHANNELS
)
_MOVE_DIM = _MOVE_SCALARS + len(_CATEGORIES) + len(_TYPES) + _MOVE_ID_CHANNELS
_GLOBAL_DIM = len(_WEATHERS) + len(_FIELDS) + 2 * len(_SIDE_CONDITIONS) + _GLOBAL_SCALARS

OBS_DIM = _TEAM_SIZE * 2 * _POKEMON_DIM + _N_MOVES * _MOVE_DIM + _GLOBAL_DIM


@dataclass(frozen=True)
class ObsLayout:
    """Public, drift-proof view of the flat-obs block structure.

    ``embed_battle`` concatenates blocks in a fixed order: ``team_size`` ego
    Pokemon, ``team_size`` opponent Pokemon, ``n_moves`` active moves, then 1
    global block. Consumers (e.g. the entity transformer) slice the flat vector
    back into per-entity tokens using these dims, so the token layout can never
    silently drift from the encoder.
    """

    team_size: int
    n_moves: int
    pokemon_dim: int
    move_dim: int
    global_dim: int
    pokemon_id_channels: int = 0
    move_id_channels: int = 0

    @property
    def pokemon_numeric_dim(self) -> int:
        """Numeric (non-ID) channels per Pokemon block; ID channels are the trailing tail."""
        return self.pokemon_dim - self.pokemon_id_channels

    @property
    def move_numeric_dim(self) -> int:
        """Numeric (non-ID) channels per move block; ID channels are the trailing tail."""
        return self.move_dim - self.move_id_channels

    @property
    def n_pokemon_tokens(self) -> int:
        """Ego + opponent Pokemon tokens."""
        return self.team_size * 2

    @property
    def n_tokens(self) -> int:
        """Total tokens: ego + opp Pokemon, moves, and the single global token."""
        return self.team_size * 2 + self.n_moves + 1

    @property
    def moves_start(self) -> int:
        """Flat-obs offset where the move blocks begin."""
        return self.team_size * 2 * self.pokemon_dim

    @property
    def global_start(self) -> int:
        """Flat-obs offset where the global block begins."""
        return self.moves_start + self.n_moves * self.move_dim


OBS_LAYOUT = ObsLayout(
    team_size=_TEAM_SIZE,
    n_moves=_N_MOVES,
    pokemon_dim=_POKEMON_DIM,
    move_dim=_MOVE_DIM,
    global_dim=_GLOBAL_DIM,
    pokemon_id_channels=_POKEMON_ID_CHANNELS,
    move_id_channels=_MOVE_ID_CHANNELS,
)


_T = TypeVar("_T")


def _padded(items: Iterable[_T], size: int) -> list[_T | None]:
    """Truncate/pad ``items`` to exactly ``size`` entries, padding with ``None``."""
    out: list[_T | None] = []
    out.extend(list(items)[:size])
    out.extend([None] * (size - len(out)))
    return out


def _one_hot(index: int | None, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    if index is not None:
        vec[index] = 1.0
    return vec


def _encode_pokemon(mon: Pokemon | None) -> np.ndarray:
    if mon is None:
        return np.zeros(_POKEMON_DIM, dtype=np.float32)

    flags = np.array(
        [1.0, float(bool(mon.active)), float(mon.fainted), float(mon.current_hp_fraction)],
        dtype=np.float32,
    )
    types = np.zeros(len(_TYPES), dtype=np.float32)
    for t in mon.types:
        if t is not None:
            types[_TYPE_INDEX[t]] = 1.0
    status = _one_hot(_STATUS_INDEX.get(mon.status) if mon.status else None, len(_STATUSES))
    boosts = np.array(
        [mon.boosts.get(k, 0) / _BOOST_SCALE for k in _BOOST_KEYS], dtype=np.float32
    )
    base_stats = np.array(
        [mon.base_stats.get(k, 0) / _STAT_SCALE for k in _STAT_KEYS], dtype=np.float32
    )
    # Trailing ID channels, in POKEMON_ID_FIELDS order (species, item, ability).
    ids = np.array(
        [vocab.species_id(mon), vocab.item_id(mon), vocab.ability_id(mon)], dtype=np.float32
    )
    return np.concatenate([flags, types, status, boosts, base_stats, ids])


def _encode_move(
    move: Move | None, attacker: Pokemon | None, defender: Pokemon | None
) -> np.ndarray:
    if move is None:
        return np.zeros(_MOVE_DIM, dtype=np.float32)

    max_pp = move.max_pp or 1
    score = score_move(move, attacker, defender) if attacker is not None else 0.0
    scalars = np.array(
        [
            1.0,
            (move.base_power or 0) / _BP_SCALE,
            normalize_accuracy(move.accuracy),
            move.priority / _PRIORITY_SCALE,
            move.current_pp / max_pp,
            score / _SCORE_SCALE,
        ],
        dtype=np.float32,
    )
    category = _one_hot(_CATEGORY_INDEX.get(move.category), len(_CATEGORIES))
    move_type = _one_hot(_TYPE_INDEX.get(move.type) if move.type else None, len(_TYPES))
    # Trailing ID channel, in MOVE_ID_FIELDS order (move).
    ids = np.array([vocab.move_id(move)], dtype=np.float32)
    return np.concatenate([scalars, category, move_type, ids])


def _presence(active: dict, members: list) -> np.ndarray:
    """One feature per enum member: stack fraction if stackable, else presence."""
    vec = np.zeros(len(members), dtype=np.float32)
    for i, member in enumerate(members):
        if member in active:
            cap = STACKABLE_CONDITIONS.get(member)
            vec[i] = active[member] / cap if cap else 1.0
    return vec


def _encode_global(battle: AbstractBattle) -> np.ndarray:
    weather = _presence(battle.weather, _WEATHERS)
    fields = _presence(battle.fields, _FIELDS)
    own_side = _presence(battle.side_conditions, _SIDE_CONDITIONS)
    opp_side = _presence(battle.opponent_side_conditions, _SIDE_CONDITIONS)
    active = battle.active_pokemon
    flags = np.array(
        [
            battle.turn / _TURN_SCALE,
            float(battle.force_switch),
            float(bool(battle.can_tera)),
            float(battle.maybe_trapped),
            # Own active's first turn since switch-in (Build 9 Gate B): an explicit,
            # in-distribution recency signal for "just switched, haven't acted". Build 8 showed
            # the switch loop leaked this through pp (all moves at full pp == never attacked, an
            # off-manifold cue); representing "just switched" honestly lets the policy read it
            # without that pp artifact. Driven by the same protocol messages offline and live,
            # so -- unlike a game-depth counter -- it reconstructs identically on both sides.
            float(active.first_turn) if active is not None else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([weather, fields, own_side, opp_side, flags])


def _opponent_mons(battle: AbstractBattle) -> list[Pokemon]:
    """Opponent Pokemon to encode: revealed (detailed) first, then team-preview species.

    In team-choice formats a real player sees all six opponent species from team preview; poke-env
    keeps the not-yet-revealed ones in ``teampreview_opponent_team`` (separate from the revealed
    ``opponent_team``). Merging them -- revealed first, preview filling the rest -- makes the
    encoded opponent roster identical at offline reconstruction (``data.ingest``) and live, same
    order by construction, so the policy is never trained on more than it sees at eval. This is a
    no-op for random battles (no team preview, so ``teampreview_opponent_team`` is empty)."""
    revealed = list(battle.opponent_team.values())
    seen = {m.species for m in revealed}
    teampreview = getattr(battle, "teampreview_opponent_team", []) or []
    preview = [m for m in teampreview if m.species not in seen]
    return revealed + preview


def embed_battle(battle: AbstractBattle) -> np.ndarray:
    """Encode ``battle`` (our POV) into a fixed-size ``float32`` vector of ``OBS_DIM``."""
    team = _padded(battle.team.values(), _TEAM_SIZE)
    opp_team = _padded(_opponent_mons(battle), _TEAM_SIZE)

    blocks = [_encode_pokemon(m) for m in team]
    blocks += [_encode_pokemon(m) for m in opp_team]

    active = battle.active_pokemon
    defender = battle.opponent_active_pokemon
    move_values = canonical_moves(active) if active is not None else []
    moves = _padded(move_values, _N_MOVES)
    blocks += [_encode_move(m, active, defender) for m in moves]

    blocks.append(_encode_global(battle))

    obs = np.concatenate(blocks)
    assert obs.shape == (OBS_DIM,), f"encoder produced {obs.shape}, expected {(OBS_DIM,)}"
    return obs

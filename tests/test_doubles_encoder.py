"""The doubles observation, and the one thing that must NOT change while adding it.

G4 claims a third format plugs in "without rewriting the core". The encoder is where that is
cheapest to violate: touching the shared layout would silently invalidate every singles checkpoint
and every collected shard, because both are fingerprinted by (OBS_VERSION, OBS_DIM). So the first
test here is a freeze on singles, and the doubles layout is a separate, separately-versioned one.

What genuinely differs is only the move blocks (per active slot rather than per battle) and the
global block (per-slot force_switch / trapped / first_turn, plus which foe slots are occupied --
the targeting context a factored per-slot head needs to pick a target at all).
"""

from __future__ import annotations

import numpy as np

from lategame.features.doubles_encoder import (
    N_ACTIVE,
    OBS_DIM_DOUBLES,
    OBS_LAYOUT_DOUBLES,
    OBS_VERSION_DOUBLES,
    embed_doubles_battle,
)
from lategame.features.encoder import OBS_DIM, OBS_LAYOUT, OBS_VERSION


def test_the_singles_layout_is_frozen():
    """Every existing checkpoint and shard is fingerprinted by (version, dim). Adding doubles must
    not move either, or the whole singles history silently fails to load."""
    assert OBS_DIM == 761
    assert OBS_VERSION.startswith("v5-")
    assert OBS_LAYOUT.n_active == 1
    assert OBS_LAYOUT.n_move_blocks == OBS_LAYOUT.n_moves  # one active => unchanged arithmetic
    assert OBS_LAYOUT.global_start == (
        OBS_LAYOUT.moves_start + OBS_LAYOUT.n_moves * OBS_LAYOUT.move_dim
    )


def test_the_doubles_layout_is_separately_versioned():
    """A doubles shard must never be loadable by a singles run. Different version AND dim, so the
    fingerprint check in data.collect rejects it on either field alone."""
    assert OBS_VERSION_DOUBLES != OBS_VERSION
    assert OBS_DIM_DOUBLES != OBS_DIM
    assert OBS_LAYOUT_DOUBLES.n_active == N_ACTIVE == 2


def test_doubles_emits_move_blocks_per_active_slot():
    """The substantive difference: a policy has to score BOTH slots' options, so the obs carries
    n_active x n_moves move blocks rather than one set for 'the' active."""
    assert OBS_LAYOUT_DOUBLES.n_move_blocks == 2 * OBS_LAYOUT_DOUBLES.n_moves == 8
    assert OBS_LAYOUT_DOUBLES.global_start == (
        OBS_LAYOUT_DOUBLES.moves_start + 8 * OBS_LAYOUT_DOUBLES.move_dim
    )
    assert OBS_LAYOUT_DOUBLES.n_tokens == OBS_LAYOUT.n_tokens + 4  # 4 extra move tokens


def test_the_layout_arithmetic_closes():
    """global_start + global_dim must land exactly on OBS_DIM, or the entity transformer slices
    tokens off the end of the vector."""
    layout = OBS_LAYOUT_DOUBLES
    assert layout.global_start + layout.global_dim == OBS_DIM_DOUBLES
    assert OBS_LAYOUT.global_start + OBS_LAYOUT.global_dim == OBS_DIM


def test_a_live_doubles_pov_encodes_to_the_right_shape(monkeypatch):
    """Against a real DoubleBattle-shaped POV rather than a hand-built vector."""
    from tests.test_doubles_action_space import _battle

    obs = embed_doubles_battle(_battle())
    assert obs.shape == (OBS_DIM_DOUBLES,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert obs.any(), "a battle with two healthy actives must not encode to all zeros"


def test_an_empty_slot_encodes_as_zeros_not_as_a_crash():
    """A fainted-and-not-yet-replaced slot is the ordinary mid-battle case on doubles."""
    from tests.test_doubles_action_space import _battle

    b = _battle()
    b._a = [b._a[0], None]  # slot 1 empty
    obs = embed_doubles_battle(b)
    assert obs.shape == (OBS_DIM_DOUBLES,)
    # The "our slot 1 occupied" flag is in the global block; it must read 0 now and 1 when filled.
    occupied_flag = obs[OBS_LAYOUT_DOUBLES.global_start + OBS_LAYOUT_DOUBLES.global_dim - 3]
    assert occupied_flag == 0.0

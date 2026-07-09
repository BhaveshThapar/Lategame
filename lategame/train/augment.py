"""Train-time feature augmentation for BC (plan.md 13, Build 10).

The Gen 9 OU switch loop is a *joint out-of-distribution* input: the looping agent
never attacks, so all four of the active mon's moves stay at *exactly* full pp deep
into a game -- a corner that never occurs in winning human play (a winner has spent
pp by mid-game). The policy extrapolates arbitrarily there and reads full-pp as a
switch cue (Build 8/9: neutralizing pp to in-distribution drops live switch mass
0.604 -> 0.244). pp is load-bearing for imitation (can't drop it, Gate A) and can't
be out-voted by a parallel feature (Gate B), so the remaining fix is to make that
corner in-distribution: synthesize "all moves full pp at a deep turn -> attack".

``augment_pp_full`` does exactly that, at train time only (validation stays clean):
on a random fraction of attack-labeled, deep-turn rows it forces the active mon's pp
channels to full. Column offsets are read from ``ObsLayout`` so they can never drift
from ``encoder.embed_battle``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lategame.features.encoder import ObsLayout

# Scalar offsets inside each move block, mirroring encoder ``_MOVE_SCALARS``
# ("present, base_power, accuracy, priority, pp_fraction, score"). Same indices the
# drift probe swaps on (``scripts/drift_probe.py`` ``_MOVE_PP_IDX = 4``).
_MOVE_PRESENT_IDX = 0
_MOVE_PP_IDX = 4


def pp_columns(layout: ObsLayout) -> list[int]:
    """Flat-obs indices of the active mon's ``pp_fraction`` channels (one per move)."""
    return [layout.moves_start + j * layout.move_dim + _MOVE_PP_IDX for j in range(layout.n_moves)]


def augment_pp_full(
    obs: Tensor,
    action: Tensor,
    layout: ObsLayout,
    *,
    frac: float,
    turn_threshold: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return a clone of ``obs`` with the active mon's pp channels forced to full (1.0)
    on a random ``frac`` of attack-labeled, deep-turn rows.

    - attack row: ``action >= layout.team_size`` (switch actions are ``0..team_size-1``).
    - deep turn: normalized turn (first global scalar) ``>= turn_threshold``.
    - pp is set to 1.0 only where the move is *present* (present flag == 1), so absent
      move blocks stay all-zero, matching the encoder.

    ``frac <= 0`` is a no-op (returns ``obs`` untouched). Never mutates ``obs`` in place.
    """
    if frac <= 0.0:
        return obs
    out = obs.clone()
    attack = action >= layout.team_size
    deep = out[:, layout.global_start] >= turn_threshold  # global scalar 0 == normalized turn
    roll = torch.rand(out.shape[0], generator=generator).to(out.device)
    sel = attack & deep & (roll < frac)
    if not bool(sel.any()):
        return out
    for j in range(layout.n_moves):
        base = layout.moves_start + j * layout.move_dim
        rows = sel & (out[:, base + _MOVE_PRESENT_IDX] > 0.0)
        out[rows, base + _MOVE_PP_IDX] = 1.0
    return out

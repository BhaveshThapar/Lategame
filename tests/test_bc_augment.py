"""Unit tests for the Build-10 train-time pp augmentation (plan.md 13).

``augment_pp_full`` forces the active mon's pp channels to full on a random fraction of
attack-labeled, deep-turn rows -- and must leave everything else (switch rows, shallow
turns, absent-move blocks, non-pp columns) untouched. Column offsets come from
``OBS_LAYOUT`` so these tests also pin the augmentation to the encoder layout.
"""

from __future__ import annotations

import torch

from lategame.features.encoder import OBS_DIM, OBS_LAYOUT
from lategame.train.augment import augment_pp_full, pp_columns

_L = OBS_LAYOUT
_TURN_COL = _L.global_start
_PP_COLS = pp_columns(_L)
_PRESENT_COLS = [_L.moves_start + j * _L.move_dim for j in range(_L.n_moves)]


def _row(*, turn: float, n_present: int, pp: float = 0.3) -> torch.Tensor:
    """One obs row: ``n_present`` active moves at ``pp`` fraction, given normalized turn."""
    obs = torch.zeros(OBS_DIM, dtype=torch.float32)
    obs[_TURN_COL] = turn
    for j in range(n_present):
        obs[_PRESENT_COLS[j]] = 1.0
        obs[_PP_COLS[j]] = pp
    return obs


def _augment(obs: torch.Tensor, action: torch.Tensor, *, frac: float = 1.0) -> torch.Tensor:
    return augment_pp_full(obs, action, _L, frac=frac, turn_threshold=0.15)


def test_attack_deep_row_sets_present_pp_full() -> None:
    obs = _row(turn=0.5, n_present=4).unsqueeze(0)
    out = _augment(obs, torch.tensor([10]))
    assert torch.allclose(out[0, _PP_COLS], torch.ones(_L.n_moves))


def test_only_pp_columns_change() -> None:
    obs = _row(turn=0.5, n_present=4).unsqueeze(0)
    out = _augment(obs, torch.tensor([10])).clone()
    out[0, _PP_COLS] = obs[0, _PP_COLS]  # revert the pp edit
    assert torch.equal(out, obs)  # ...everything else must be identical


def test_absent_move_pp_stays_zero() -> None:
    obs = _row(turn=0.5, n_present=2).unsqueeze(0)  # moves 2,3 absent (present=0, pp=0)
    out = _augment(obs, torch.tensor([10]))
    assert out[0, _PP_COLS[0]] == 1.0 and out[0, _PP_COLS[1]] == 1.0
    assert out[0, _PP_COLS[2]] == 0.0 and out[0, _PP_COLS[3]] == 0.0


def test_switch_row_untouched() -> None:
    obs = _row(turn=0.5, n_present=4).unsqueeze(0)
    out = _augment(obs, torch.tensor([3]))  # action < team_size -> switch
    assert torch.equal(out, obs)


def test_shallow_turn_untouched() -> None:
    obs = _row(turn=0.0, n_present=4).unsqueeze(0)
    out = _augment(obs, torch.tensor([10]))
    assert torch.equal(out, obs)


def test_frac_zero_is_noop() -> None:
    obs = _row(turn=0.5, n_present=4).unsqueeze(0)
    out = augment_pp_full(obs, torch.tensor([10]), _L, frac=0.0, turn_threshold=0.15)
    assert out is obs


def test_does_not_mutate_input() -> None:
    obs = _row(turn=0.5, n_present=4).unsqueeze(0)
    before = obs.clone()
    _augment(obs, torch.tensor([10]))
    assert torch.equal(obs, before)


def test_deterministic_under_seeded_generator() -> None:
    batch = torch.stack([_row(turn=0.5, n_present=4) for _ in range(64)])
    action = torch.full((64,), 10)
    a = augment_pp_full(
        batch, action, _L, frac=0.5, turn_threshold=0.15, generator=torch.Generator().manual_seed(0)
    )
    b = augment_pp_full(
        batch, action, _L, frac=0.5, turn_threshold=0.15, generator=torch.Generator().manual_seed(0)
    )
    assert torch.equal(a, b)
    # frac=0.5 actually leaves some rows unaugmented (not a degenerate all-or-nothing selection).
    n_full = int((a[:, _PP_COLS[0]] == 1.0).sum())
    assert 0 < n_full < 64

"""Unit tests for the Build-10 train-time pp augmentation (docs/RESULTS.md).

``augment_pp_full`` forces the active mon's pp channels to full on a random fraction of
attack-labeled, deep-turn rows -- and must leave everything else (switch rows, shallow
turns, absent-move blocks, non-pp columns) untouched. Column offsets come from
``OBS_LAYOUT`` so these tests also pin the augmentation to the encoder layout.
"""

from __future__ import annotations

import torch

from rotomai.features.encoder import OBS_DIM, OBS_LAYOUT
from rotomai.train.augment import (
    augment_pp_full,
    augment_pp_noise,
    augment_pp_resample,
    pp_columns,
)

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


# ---- Build 11: global pp regularizers (augment_pp_noise / augment_pp_resample) ----
# Unlike augment_pp_full these apply in EVERY context (switch rows, shallow turns) -- the tests
# below pin that global reach and the [0,1] / in-distribution range invariants.


def _noise(obs: torch.Tensor, action: torch.Tensor, *, std: float = 0.2) -> torch.Tensor:
    return augment_pp_noise(obs, action, _L, std=std, generator=torch.Generator().manual_seed(0))


def _resample(obs: torch.Tensor, action: torch.Tensor, *, frac: float = 1.0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    return augment_pp_resample(obs, action, _L, frac=frac, generator=gen)


def test_noise_perturbs_present_pp_globally() -> None:
    # Switch action AND shallow turn -> augment_pp_full skips both, but the global noise must not.
    obs = torch.stack([_row(turn=0.0, n_present=4, pp=0.3) for _ in range(8)])
    out = _noise(obs, torch.zeros(8, dtype=torch.long))  # all switch actions
    assert not torch.equal(out[:, _PP_COLS], obs[:, _PP_COLS])


def test_noise_stays_in_unit_range() -> None:
    obs = torch.stack([_row(turn=0.5, n_present=4, pp=0.98) for _ in range(64)])
    out = _noise(obs, torch.full((64,), 10), std=0.5)
    pp = out[:, _PP_COLS]
    assert float(pp.min()) >= 0.0 and float(pp.max()) <= 1.0


def test_noise_only_pp_columns_change() -> None:
    obs = torch.stack([_row(turn=0.5, n_present=4, pp=0.3) for _ in range(8)])
    out = _noise(obs, torch.full((8,), 10)).clone()
    out[:, _PP_COLS] = obs[:, _PP_COLS]  # revert the pp edits
    assert torch.equal(out, obs)


def test_noise_absent_move_pp_stays_zero() -> None:
    obs = _row(turn=0.5, n_present=2, pp=0.3).unsqueeze(0)  # moves 2,3 absent
    out = _noise(obs, torch.tensor([10]), std=0.5)
    assert out[0, _PP_COLS[2]] == 0.0 and out[0, _PP_COLS[3]] == 0.0


def test_noise_zero_std_is_noop() -> None:
    obs = _row(turn=0.5, n_present=4).unsqueeze(0)
    out = augment_pp_noise(obs, torch.tensor([10]), _L, std=0.0)
    assert out is obs


def test_noise_does_not_mutate_input() -> None:
    obs = torch.stack([_row(turn=0.5, n_present=4, pp=0.3) for _ in range(8)])
    before = obs.clone()
    _noise(obs, torch.full((8,), 10))
    assert torch.equal(obs, before)


def test_noise_deterministic_under_seeded_generator() -> None:
    batch = torch.stack([_row(turn=0.5, n_present=4, pp=0.3) for _ in range(32)])
    action = torch.full((32,), 10)
    a = augment_pp_noise(batch, action, _L, std=0.2, generator=torch.Generator().manual_seed(1))
    b = augment_pp_noise(batch, action, _L, std=0.2, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


def test_resample_perturbs_present_pp_globally() -> None:
    pps = torch.linspace(0.1, 0.9, 16)
    obs = torch.stack([_row(turn=0.0, n_present=4, pp=float(p)) for p in pps])
    out = _resample(obs, torch.zeros(16, dtype=torch.long))  # switch + shallow rows
    assert not torch.equal(out[:, _PP_COLS], obs[:, _PP_COLS])


def test_resample_values_come_from_pool() -> None:
    obs = torch.stack([_row(turn=0.5, n_present=4, pp=p) for p in (0.2, 0.4, 0.6, 0.8)])
    pool = {0.2, 0.4, 0.6, 0.8}  # every present-move pp value in the input
    out = _resample(obs, torch.full((4,), 10))
    assert all(round(v, 4) in pool for v in out[:, _PP_COLS].flatten().tolist())


def test_resample_absent_move_pp_stays_zero() -> None:
    obs = _row(turn=0.5, n_present=2, pp=0.3).unsqueeze(0)
    out = _resample(obs, torch.tensor([10]))
    assert out[0, _PP_COLS[2]] == 0.0 and out[0, _PP_COLS[3]] == 0.0


def test_resample_zero_frac_is_noop() -> None:
    obs = _row(turn=0.5, n_present=4).unsqueeze(0)
    out = augment_pp_resample(obs, torch.tensor([10]), _L, frac=0.0)
    assert out is obs


def test_resample_does_not_mutate_input() -> None:
    obs = torch.stack([_row(turn=0.5, n_present=4, pp=0.3) for _ in range(8)])
    before = obs.clone()
    _resample(obs, torch.full((8,), 10))
    assert torch.equal(obs, before)


def test_resample_deterministic_under_seeded_generator() -> None:
    pps = torch.linspace(0.1, 0.9, 32)
    batch = torch.stack([_row(turn=0.5, n_present=4, pp=float(p)) for p in pps])
    action = torch.full((32,), 10)
    a = augment_pp_resample(batch, action, _L, frac=0.5, generator=torch.Generator().manual_seed(2))
    b = augment_pp_resample(batch, action, _L, frac=0.5, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a, b)

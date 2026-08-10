"""Resume must continue a run, not restart one -- and must be invisible when off.

Build 26 needs this because arm length is bounded by MEMORY, not walltime: RSS grows ~0.23
GiB/iter and the medium QoS caps a job at 64 GB per job, so v26b was OUT_OF_MEMORY at 304/320.
A fresh process resets RSS, which makes resume the only way to run an arm past ~260 iters.

That puts real weight on two properties. (1) With ``resume`` off, nothing changes -- no state is
written and no state is read, so every build through 25 still reproduces. (2) With it on, the
restart picks up Adam's moments and both RNG streams, because dropping them would restart the
optimizer cold mid-arm and re-draw the league from the top of the stream -- a discontinuity in
training dynamics dressed up as a bookkeeping detail.
"""

import random

import pytest
import torch

from lategame.train.ppo import (
    _RESUME_FILE,
    _load_resume_state,
    _save_resume_state,
)


def _optimizer(seed: int = 0) -> torch.optim.Optimizer:
    torch.manual_seed(seed)
    param = torch.nn.Parameter(torch.randn(8, 4))
    opt = torch.optim.Adam([param], lr=2.5e-4)
    # Take a step so Adam actually has moment estimates to lose.
    (param.sum()).backward()
    opt.step()
    return opt


def test_returns_none_when_there_is_nothing_to_resume(tmp_path):
    assert _load_resume_state(tmp_path, iters=320) is None


def test_round_trips_optimizer_moments_rng_and_curve(tmp_path):
    opt = _optimizer()
    rng = random.Random(7)
    rng.random()  # advance off the seed so a naive re-seed would be detectable
    curve = [{"iter": 0, "vs_heuristic": 0.1}, {"iter": 1, "vs_heuristic": 0.2}]
    (tmp_path / "iter_01.pt").write_bytes(b"stub")

    _save_resume_state(tmp_path / _RESUME_FILE, 1, opt, curve, rng)
    state = _load_resume_state(tmp_path, iters=320)

    assert state is not None
    assert state["iteration"] == 1
    assert state["checkpoint"] == str(tmp_path / "iter_01.pt")
    assert state["curve"] == curve

    # Adam's exp_avg / exp_avg_sq survive -- the whole point.
    saved = state["optimizer"]["state"][0]
    live = opt.state_dict()["state"][0]
    assert torch.allclose(saved["exp_avg"], live["exp_avg"])
    assert torch.allclose(saved["exp_avg_sq"], live["exp_avg_sq"])

    # And the python RNG continues the stream rather than replaying it.
    restored = random.Random(0)
    restored.setstate(state["py_rng"])
    assert restored.random() == rng.random()


def test_a_torn_write_cannot_poison_the_retry(tmp_path):
    """Saved via a tmp file + atomic replace, so a kill mid-write leaves the old state intact."""
    opt = _optimizer()
    (tmp_path / "iter_01.pt").write_bytes(b"stub")
    (tmp_path / "iter_02.pt").write_bytes(b"stub")
    _save_resume_state(tmp_path / _RESUME_FILE, 1, opt, [], random.Random(0))
    _save_resume_state(tmp_path / _RESUME_FILE, 2, opt, [], random.Random(0))

    assert not (tmp_path / _RESUME_FILE).with_suffix(".tmp").exists()
    assert _load_resume_state(tmp_path, iters=320)["iteration"] == 2


def test_refuses_when_the_named_checkpoint_is_gone(tmp_path):
    """Silently starting fresh would burn the whole walltime re-running existing work."""
    _save_resume_state(tmp_path / _RESUME_FILE, 5, _optimizer(), [], random.Random(0))
    with pytest.raises(FileNotFoundError, match="iter_05.pt"):
        _load_resume_state(tmp_path, iters=320)


def test_refuses_when_the_run_is_already_complete(tmp_path):
    (tmp_path / "iter_320.pt").write_bytes(b"stub")
    _save_resume_state(tmp_path / _RESUME_FILE, 320, _optimizer(), [], random.Random(0))
    with pytest.raises(ValueError, match="nothing to resume"):
        _load_resume_state(tmp_path, iters=320)


def test_resume_defaults_off_so_build_25_is_unchanged():
    from lategame.train.ppo import PPOConfig

    assert PPOConfig().resume is False


async def test_resume_refuses_an_unpinned_anneal_horizon(tmp_path):
    """``anneal_horizon`` falls back to ``iters``, which a chunked run RAISES between chunks.

    Left unguarded, chunk 1 would anneal over 160 and chunk 2 over 320 -- the arm quietly stops
    being one experiment. This is the confound Build 24 added ``anneal_iters`` to remove.
    """
    from lategame.train.ppo import PPOConfig, run_ppo

    init = tmp_path / "init.pt"
    init.write_bytes(b"stub")
    cfg = PPOConfig(init=str(init), out_dir=str(tmp_path / "out"), iters=4, resume=True)
    with pytest.raises(ValueError, match="anneal-iters"):
        await run_ppo(cfg)


async def test_pinned_anneal_horizon_is_stable_across_chunks():
    """The property the guard buys: same horizon whatever ``iters`` a chunk was given."""
    from lategame.train.ppo import PPOConfig

    chunk1 = PPOConfig(iters=160, anneal_iters=80, resume=True)
    chunk2 = PPOConfig(iters=320, anneal_iters=80, resume=True)
    assert chunk1.anneal_horizon == chunk2.anneal_horizon == 80

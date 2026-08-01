"""Unit tests for the M4 self-play building blocks (no server needed).

Covers the league sampler, the checkpoint-kwarg forwarding shared by ``build_player``
and the recording-player builder, the replay-buffer shard concatenation, and the
actor-critic -> actor-critic warm-start. The full loop is exercised by the
server-gated ``test_selfplay_smoke``.
"""

import random

import numpy as np
import pytest

from lategame.data import collect
from lategame.data.collect import PlayerSpec, TrajectoryDataset, concat_rl_shards, save_rl
from lategame.data.reward import RewardWeights
from lategame.eval import arena
from lategame.features.action_space import GEN9_ACTION_SPACE_SIZE
from lategame.features.encoder import OBS_DIM
from lategame.train.selfplay import _sample_league


def test_sample_league_uniform_subset():
    league = ["a", "b", "c", "d"]
    picked = _sample_league(league, 2, random.Random(0))
    assert len(picked) == 2
    assert len(set(picked)) == 2  # distinct members
    assert set(picked) <= set(league)


def test_sample_league_handles_single_member():
    assert _sample_league(["only"], 3, random.Random(0)) == ["only"]


def test_build_player_forwards_checkpoint_kwargs(monkeypatch):
    captured: dict = {}

    class Dummy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(arena.AGENTS, "offrl", Dummy)
    arena.build_player("offrl", "gen9randombattle", checkpoint_path="x.pt", sample=True)
    assert captured["checkpoint_path"] == "x.pt"
    assert captured["sample"] is True


def test_build_player_omits_kwargs_for_baseline(monkeypatch):
    captured: dict = {}

    class Dummy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(arena.AGENTS, "random", Dummy)
    arena.build_player("random", "gen9randombattle", checkpoint_path="x.pt", sample=True)
    assert "checkpoint_path" not in captured
    assert "sample" not in captured


def test_recording_player_forwards_checkpoint_kwargs(monkeypatch):
    captured: dict = {}

    class Dummy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(collect.AGENTS, "offrl", Dummy)
    collect._build_recording_player(
        PlayerSpec("offrl", checkpoint_path="x.pt", sample=True), "gen9randombattle"
    )
    assert captured["checkpoint_path"] == "x.pt"
    assert captured["sample"] is True


def _tiny_traj(n: int, val: float = 1.0) -> TrajectoryDataset:
    return TrajectoryDataset(
        obs=np.zeros((n, OBS_DIM), dtype=np.float32),
        action=np.zeros(n, dtype=np.int64),
        mask=np.ones((n, GEN9_ACTION_SPACE_SIZE), dtype=bool),
        reward=np.full(n, val, dtype=np.float32),
        done=np.array([False] * (n - 1) + [True]),
        battle_format="gen9randombattle",
        gamma=1.0,
        weights=RewardWeights(),
    )


def test_concat_rl_shards_preserves_episodes(tmp_path):
    s1, s2 = tmp_path / "a.npz", tmp_path / "b.npz"
    save_rl(_tiny_traj(2), s1)
    save_rl(_tiny_traj(3), s2)
    ds = concat_rl_shards([s1, s2], tmp_path / "buf.npz")
    assert len(ds) == 5
    assert int(ds.done.sum()) == 2  # one terminal per shard, preserved across the seam


def test_concat_rl_shards_returns_reset_across_seam(tmp_path):
    pytest.importorskip("torch")
    from lategame.data.rl_dataset import RLDataset

    s1, s2 = tmp_path / "a.npz", tmp_path / "b.npz"
    save_rl(_tiny_traj(2), s1)
    save_rl(_tiny_traj(3), s2)
    out = tmp_path / "buf.npz"
    concat_rl_shards([s1, s2], out)
    # gamma=1: shard1 returns [2,1], shard2 [3,2,1] -- the scan must reset at the seam.
    np.testing.assert_allclose(RLDataset(out).ret.numpy(), [2, 1, 3, 2, 1])


def test_load_actor_critic_weights_roundtrip():
    torch = pytest.importorskip("torch")
    from lategame.model.actor_critic import ActorCritic, load_actor_critic_weights

    src = ActorCritic(OBS_DIM, hidden_dim=16, n_bins=11)
    dst = ActorCritic(OBS_DIM, hidden_dim=16, n_bins=11)
    load_actor_critic_weights(dst, src.state_dict())
    for key, tensor in src.state_dict().items():
        assert torch.equal(tensor, dst.state_dict()[key])


def test_load_actor_critic_weights_shape_mismatch_raises():
    pytest.importorskip("torch")
    from lategame.model.actor_critic import ActorCritic, load_actor_critic_weights

    src = ActorCritic(OBS_DIM, hidden_dim=16, n_bins=11)
    dst = ActorCritic(OBS_DIM, hidden_dim=16, n_bins=21)  # value head shape differs
    with pytest.raises(RuntimeError):
        load_actor_critic_weights(dst, src.state_dict())

"""Unit tests for the offline-RL dataset loader + MC return computation.

Skipped if torch is not installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lategame.data.rl_dataset import RLDataset, discounted_returns  # noqa: E402
from lategame.features.action_space import GEN9_ACTION_SPACE_SIZE  # noqa: E402
from lategame.features.encoder import OBS_DIM, OBS_VERSION  # noqa: E402


def test_discounted_returns_single_episode():
    reward = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    done = np.array([False, False, True])
    ret = discounted_returns(reward, done, gamma=0.5)
    np.testing.assert_allclose(ret, [1.75, 1.5, 1.0], rtol=1e-6)


def test_discounted_returns_resets_per_episode():
    reward = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    done = np.array([False, True, False, True])
    ret = discounted_returns(reward, done, gamma=1.0)
    np.testing.assert_allclose(ret, [2.0, 1.0, 2.0, 1.0], rtol=1e-6)


def _write_shard(path, n=4, obs_version=OBS_VERSION):
    np.savez_compressed(
        path,
        obs=np.zeros((n, OBS_DIM), dtype=np.float32),
        action=np.zeros(n, dtype=np.int64),
        mask=np.ones((n, GEN9_ACTION_SPACE_SIZE), dtype=bool),
        reward=np.ones(n, dtype=np.float32),
        done=np.array([False] * (n - 1) + [True]),
        obs_version=np.array(obs_version),
        obs_dim=np.array(OBS_DIM),
        battle_format=np.array("gen9randombattle"),
        gamma=np.array(1.0, dtype=np.float32),
    )


def test_rl_dataset_loads_and_computes_returns(tmp_path):
    path = tmp_path / "rl.npz"
    _write_shard(path, n=3)
    ds = RLDataset(path)
    assert len(ds) == 3
    obs, action, mask, ret = ds[0]
    assert obs.shape == (OBS_DIM,)
    assert mask.shape == (GEN9_ACTION_SPACE_SIZE,)
    # gamma=1, three rewards of 1.0 in one episode -> returns 3,2,1.
    np.testing.assert_allclose(ds.ret.numpy(), [3.0, 2.0, 1.0], rtol=1e-6)


def test_rl_dataset_rejects_stale_encoder(tmp_path):
    path = tmp_path / "stale.npz"
    _write_shard(path, obs_version="v0-stale")
    with pytest.raises(ValueError):
        RLDataset(path)

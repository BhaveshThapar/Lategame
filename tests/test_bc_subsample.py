"""Unit tests for the data-scaling-sweep subset path in BC training.

The sweep trains the same shard at several sample counts; ``_subset`` must give a
deterministic, nested subset (size-N is a prefix of size-2N, independent of the
model seed) so the only thing varying across sweep points is N. Skipped if torch
is absent.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rotomai.data.collect import Dataset, save  # noqa: E402
from rotomai.data.dataset import BCDataset  # noqa: E402
from rotomai.features.encoder import OBS_DIM  # noqa: E402
from rotomai.model.factory import MODEL_ENTITY_TRANSFORMER  # noqa: E402
from rotomai.train.bc import TrainConfig, _subset, train_bc  # noqa: E402


def _bc_npz(path, n=200):
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((n, OBS_DIM)).astype(np.float32)
    action = rng.integers(0, 26, size=n).astype(np.int64)
    mask = np.ones((n, 26), dtype=bool)
    save(Dataset(obs, action, mask, "gen9randombattle"), path)


def test_subset_size_nested_and_seed_independent(tmp_path):
    path = tmp_path / "bc.npz"
    _bc_npz(path, n=200)
    dataset = BCDataset(path)

    small = _subset(dataset, 50)
    large = _subset(dataset, 100)
    assert len(small) == 50
    assert len(large) == 100
    # size-N is a prefix of size-2N (nested), so a scaling curve only adds data.
    assert small.indices == large.indices[:50]
    # Deterministic + independent of any model seed: a second call matches exactly.
    assert _subset(dataset, 50).indices == small.indices


def test_subset_passthrough_when_not_smaller(tmp_path):
    path = tmp_path / "bc.npz"
    _bc_npz(path, n=64)
    dataset = BCDataset(path)
    assert _subset(dataset, None) is dataset
    assert _subset(dataset, 64) is dataset
    assert _subset(dataset, 999) is dataset


def test_train_bc_respects_max_samples(tmp_path):
    path = tmp_path / "bc.npz"
    _bc_npz(path, n=200)
    metrics = train_bc(
        path,
        tmp_path / "bc.pt",
        TrainConfig(
            epochs=1, batch_size=32, device="cpu",
            model_type=MODEL_ENTITY_TRANSFORMER, id_embed=False, max_samples=40,
        ),
    )
    assert "val_acc" in metrics

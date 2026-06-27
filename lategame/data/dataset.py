"""Torch ``Dataset`` over a collected ``.npz`` shard.

Reward filtering already happened at collection time (only winners were written),
so this just loads tensors and yields ``(obs, action, mask)``. It asserts the
shard's encoder version/dim match the live ``features.encoder`` so a model can
never be trained on a stale feature layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lategame.features.encoder import OBS_DIM, OBS_VERSION


class BCDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        data = np.load(Path(path), allow_pickle=False)
        version = str(data["obs_version"].item())
        dim = int(data["obs_dim"].item())
        if version != OBS_VERSION or dim != OBS_DIM:
            raise ValueError(
                f"Dataset encoder mismatch: shard is {version}/dim {dim}, "
                f"encoder is {OBS_VERSION}/dim {OBS_DIM}. Re-collect the data."
            )
        self.obs = torch.from_numpy(data["obs"]).float()
        self.action = torch.from_numpy(data["action"]).long()
        self.mask = torch.from_numpy(data["mask"]).bool()
        self.battle_format = str(data["battle_format"].item())

    def __len__(self) -> int:
        return self.action.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.obs[index], self.action[index], self.mask[index]

"""Torch ``Dataset`` over a collected ``.npz`` shard.

Reward filtering already happened at collection time (only winners were written),
so this just loads tensors and yields ``(obs, action, mask)``. It asserts the shard's encoder
version/dim match the codec FOR THAT SHARD'S OWN FORMAT, so a model can never be trained on a
stale feature layout -- or on a doubles shard read as though it were singles, which would be
silently meaningless rather than an error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rotomai.features.codec import codec_for


class BCDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        data = np.load(Path(path), allow_pickle=False)
        version = str(data["obs_version"].item())
        dim = int(data["obs_dim"].item())
        battle_format = str(data["battle_format"].item())
        codec = codec_for(battle_format)
        if version != codec.obs_version or dim != codec.obs_dim:
            raise ValueError(
                f"Dataset encoder mismatch: shard is {version}/dim {dim}, but the {codec.name} "
                f"codec for {battle_format!r} is {codec.obs_version}/dim {codec.obs_dim}. "
                "Re-collect the data."
            )
        self.obs = torch.from_numpy(data["obs"]).float()
        self.action = torch.from_numpy(data["action"]).long()
        self.mask = torch.from_numpy(data["mask"]).bool()
        self.battle_format = battle_format
        self.codec = codec

    def __len__(self) -> int:
        return self.action.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.obs[index], self.action[index], self.mask[index]

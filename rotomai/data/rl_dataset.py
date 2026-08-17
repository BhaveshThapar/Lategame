"""Torch ``Dataset`` over a collected offline-RL ``.npz`` shard (plan.md, M3).

Loads ``(obs, action, mask, reward, done)`` and precomputes the **discounted
Monte-Carlo return** for every turn -- scanning backward within each episode and
resetting on ``done`` -- so the AWR trainer can read a ready target. No TD
bootstrapping, so the return is exact given the trajectory.

``reward``/``done`` are kept on the object for a future IQL/TD upgrade; the AWR
path only consumes ``ret``. As with ``BCDataset`` the shard's encoder version/dim
are asserted against the live ``features.encoder`` so a stale layout can't slip in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rotomai.features.codec import codec_for


def discounted_returns(reward: np.ndarray, done: np.ndarray, gamma: float) -> np.ndarray:
    """Per-turn discounted MC return, reset at each episode boundary (``done``)."""
    ret = np.empty_like(reward, dtype=np.float32)
    running = 0.0
    for i in range(len(reward) - 1, -1, -1):
        if done[i]:
            running = 0.0
        running = float(reward[i]) + gamma * running
        ret[i] = running
    return ret


class RLDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        data = np.load(Path(path), allow_pickle=False)
        version = str(data["obs_version"].item())
        dim = int(data["obs_dim"].item())
        battle_format = str(data["battle_format"].item())
        codec = codec_for(battle_format)
        if version != codec.obs_version or dim != codec.obs_dim:
            raise ValueError(
                f"Dataset encoder mismatch: shard is {version}/dim {dim}, "
                f"the {codec.name} codec for {battle_format!r} is {codec.obs_version}/dim "
                f"{codec.obs_dim}. Re-collect the data."
            )
        self.gamma = float(data["gamma"].item())
        self.battle_format = str(data["battle_format"].item())
        self.obs = torch.from_numpy(data["obs"]).float()
        self.action = torch.from_numpy(data["action"]).long()
        self.mask = torch.from_numpy(data["mask"]).bool()
        self.reward = torch.from_numpy(data["reward"]).float()
        self.done = torch.from_numpy(data["done"]).bool()
        ret = discounted_returns(data["reward"], data["done"], self.gamma)
        self.ret = torch.from_numpy(ret).float()

    def __len__(self) -> int:
        return self.action.shape[0]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.obs[index], self.action[index], self.mask[index], self.ret[index]

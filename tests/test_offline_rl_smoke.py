"""End-to-end offline-RL pipeline smoke test (needs a local Showdown server on :8000).

collect full trajectories -> shaped-reward dataset -> warm-start from a BC ckpt ->
train 1 epoch -> the `offrl` agent loads the checkpoint and plays a full battle.
Skipped automatically when the server is not running.
"""

import socket

import pytest

from lategame.data.collect import collect_trajectories, save_rl
from lategame.data.rl_dataset import RLDataset
from lategame.eval.arena import evaluate
from lategame.features.encoder import OBS_DIM


def _server_up(host: str = "localhost", port: int = 8000, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _server_up(), reason="local Showdown server not running on :8000"
)


def _write_bc_checkpoint(path, hidden_dim=32):
    import torch

    from lategame.features.encoder import OBS_VERSION
    from lategame.model.policy import BCPolicy

    model = BCPolicy(OBS_DIM, hidden_dim=hidden_dim)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": OBS_DIM,
            "hidden_dim": hidden_dim,
            "n_actions": model.n_actions,
            "obs_version": OBS_VERSION,
            "battle_format": "gen9randombattle",
            "metrics": {},
        },
        path,
    )


async def test_offline_rl_pipeline_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from lategame.agents.offline_rl_agent import CHECKPOINT_ENV_VAR
    from lategame.train.offline_rl import OfflineRLConfig, train_offline_rl

    # 1. Collect a tiny full-trajectory dataset (wins AND losses, shaped rewards).
    dataset = await collect_trajectories(["random", "maxbasepower"], n_per_pair=2)
    assert len(dataset) > 0
    assert dataset.obs.shape[1] == OBS_DIM
    assert dataset.done.any()  # at least one terminal turn
    assert all(dataset.mask[i, dataset.action[i]] for i in range(len(dataset)))

    data_path = tmp_path / "rl.npz"
    save_rl(dataset, data_path)
    assert len(RLDataset(data_path)) == len(dataset)

    # 2. Warm-start from a (fresh) BC checkpoint and train one epoch on CPU.
    bc_path = tmp_path / "bc.pt"
    _write_bc_checkpoint(bc_path, hidden_dim=32)
    ckpt_path = tmp_path / "offrl.pt"
    train_offline_rl(
        data_path,
        ckpt_path,
        OfflineRLConfig(epochs=1, batch_size=32, hidden_dim=32, device="cpu", bc_init=str(bc_path)),
    )
    assert ckpt_path.exists()

    # 3. The `offrl` agent loads that checkpoint (via env var) and plays a real battle.
    monkeypatch.setenv(CHECKPOINT_ENV_VAR, str(ckpt_path))
    result = await evaluate("offrl", "random", n_battles=1)
    assert result.n_battles == 1
    assert 0.0 <= result.p1_win_rate <= 1.0

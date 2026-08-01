"""End-to-end BC pipeline smoke test (needs a local Showdown server on :8000).

collect a few games -> reward-filtered dataset -> train 1 epoch -> the `bc`
agent loads the checkpoint and plays a full battle. Skipped automatically when
the server is not running.
"""

import socket

import pytest

from lategame.data.collect import collect
from lategame.data.collect import save as save_dataset
from lategame.data.dataset import BCDataset
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


async def test_bc_pipeline_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from lategame.agents.bc_agent import CHECKPOINT_ENV_VAR
    from lategame.train.bc import TrainConfig, train_bc

    # 1. Collect a tiny reward-filtered self-play dataset.
    dataset = await collect(["random", "maxbasepower"], n_per_pair=2)
    assert len(dataset) > 0
    assert dataset.obs.shape[1] == OBS_DIM
    # Every recorded action must be legal under its own mask (codec/mask consistency).
    assert all(dataset.mask[i, dataset.action[i]] for i in range(len(dataset)))

    data_path = tmp_path / "smoke.npz"
    save_dataset(dataset, data_path)

    # 2. Loading round-trips through the version-checked dataset.
    loaded = BCDataset(data_path)
    assert len(loaded) == len(dataset)

    # 3. Train one epoch on CPU and write a checkpoint.
    ckpt_path = tmp_path / "bc.pt"
    train_bc(data_path, ckpt_path, TrainConfig(epochs=1, batch_size=32, device="cpu"))
    assert ckpt_path.exists()

    # 4. The `bc` agent loads that checkpoint (via env var) and plays a real battle.
    monkeypatch.setenv(CHECKPOINT_ENV_VAR, str(ckpt_path))
    result = await evaluate("bc", "random", n_battles=1)
    assert result.n_battles == 1
    assert 0.0 <= result.p1_win_rate <= 1.0

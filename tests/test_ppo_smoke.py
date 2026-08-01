"""End-to-end M5 Phase-2 PPO smoke test (needs a local Showdown server on :8000).

Runs one tiny PPO iteration from a fresh actor-critic checkpoint: roll out a couple of
league+anchor games with the recording agent, run one PPO epoch, and chart win-rate --
exercising collect_rollout -> compute_gae -> ppo_update -> eval. Skipped automatically
when the server is not running.
"""

import json
import socket

import pytest

from lategame.features.encoder import OBS_DIM, OBS_VERSION


def _server_up(host: str = "localhost", port: int = 8000, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _server_up(), reason="local Showdown server not running on :8000"
)


def _write_ac_checkpoint(path, hidden_dim=32, n_bins=21, v_min=-5.0, v_max=5.0):
    import torch

    from lategame.model.actor_critic import ActorCritic

    model = ActorCritic(OBS_DIM, hidden_dim=hidden_dim, n_bins=n_bins)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": "actor_critic",
            "input_dim": OBS_DIM,
            "hidden_dim": hidden_dim,
            "n_actions": model.n_actions,
            "n_bins": n_bins,
            "v_min": v_min,
            "v_max": v_max,
            "obs_version": OBS_VERSION,
            "battle_format": "gen9randombattle",
            "metrics": {},
        },
        path,
    )


async def test_ppo_one_iteration_end_to_end(tmp_path):
    pytest.importorskip("torch")
    from lategame.train.ppo import PPOConfig, run_ppo

    init = tmp_path / "iter0.pt"
    _write_ac_checkpoint(init)

    config = PPOConfig(
        init=str(init),
        out_dir=str(tmp_path / "ckpts"),
        iters=1,
        games_per_opp=1,
        pop_size=1,
        max_concurrent=2,
        anchors=("simpleheuristics",),
        eval_baselines=("random",),
        eval_n=1,
        epochs=1,
        minibatch=8,
        device="cpu",
    )
    curve = await run_ppo(config)

    assert (tmp_path / "ckpts" / "iter_01.pt").exists()
    assert [point["iter"] for point in curve] == [0, 1]

    saved = json.loads((tmp_path / "ckpts" / "curve.json").read_text())
    assert [point["iter"] for point in saved] == [0, 1]
    assert "vs_random" in saved[1]
    assert "vs_iter0" in saved[1]

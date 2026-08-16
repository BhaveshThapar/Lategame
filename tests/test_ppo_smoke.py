"""End-to-end M5 Phase-2 PPO smoke test (needs a local Showdown server on :8000).

Runs one tiny PPO iteration from a fresh actor-critic checkpoint: roll out a couple of
league+anchor games with the recording agent, run one PPO epoch, and chart win-rate --
exercising collect_rollout -> compute_gae -> ppo_update -> eval. Skipped automatically
when the server is not running.
"""

import json

import pytest

from lategame.features.encoder import OBS_DIM
from tests.conftest import requires_server, write_ac_checkpoint

pytestmark = requires_server


async def test_ppo_one_iteration_end_to_end(tmp_path):
    pytest.importorskip("torch")
    from lategame.train.ppo import PPOConfig, run_ppo

    init = tmp_path / "iter0.pt"
    write_ac_checkpoint(init, OBS_DIM)

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

"""End-to-end M4 self-play smoke tests (need a local Showdown server on :8000).

Run one tiny self-play iteration from a fresh actor-critic checkpoint: collect a
couple of league+anchor games, fine-tune one epoch, and chart win-rate -- exercising
collect_selfplay -> concat_rl_shards -> AC->AC warm-started train_offline_rl -> eval.
Skipped automatically when the server is not running.

Two formats, because M4 on doubles is a different code path at every step and had none of this:
the singles case pins that nothing regressed, and the VGC case pins that the doubles codec, the
doubles agent dispatch, and the team pool all survive a real loop. The VGC repair was verified by
hand once and nothing re-ran it, which is how the singles-only hardcodes got there in the first
place.
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


async def test_selfplay_one_iteration_end_to_end(tmp_path):
    pytest.importorskip("torch")
    from lategame.train.selfplay import SelfPlayConfig, run_selfplay

    init = tmp_path / "iter0.pt"
    _write_ac_checkpoint(init)

    config = SelfPlayConfig(
        init=str(init),
        out_dir=str(tmp_path / "ckpts"),
        data_dir=str(tmp_path / "data"),
        iters=1,
        games_per_opp=1,
        pop_size=1,
        buffer_iters=1,
        anchors=("heuristic",),
        eval_baselines=("random",),
        eval_n=1,
        epochs=1,
        device="cpu",
    )
    curve = await run_selfplay(config)

    assert (tmp_path / "ckpts" / "iter_01.pt").exists()
    assert [point["iter"] for point in curve] == [0, 1]  # iter-0 baseline + iter-1

    saved = json.loads((tmp_path / "ckpts" / "curve.json").read_text())
    assert [point["iter"] for point in saved] == [0, 1]
    assert "vs_random" in saved[1]
    assert "vs_iter0" in saved[1]


async def test_selfplay_one_iteration_end_to_end_on_vgc(tmp_path):
    """The doubles M4 path, end to end on a real server.

    Everything here is a different object than the test above: an 888-d/214-logit codec, the
    `doubles` agent instead of `offrl`, a packed VGC team pool feeding BOTH collection and eval,
    and the doubles loop guard. The repair that made this runnable was five hardcoded agent names,
    a `team=` missing from all four of `_eval_point`'s builds, a `team_pool` parameter
    `collect_selfplay` did not have, and a warm-start check `run_selfplay` did not do -- and every
    one of those failed at a point no unit test reaches, because they only fail against a server.

    Deliberately not asserting a win rate. One battle of a random-init doubles policy is not a
    measurement, and G4's exit criterion says the same thing about its own 0.667.
    """
    pytest.importorskip("torch")
    from lategame.config import VGC_FORMAT
    from lategame.features.doubles_encoder import OBS_DIM_DOUBLES, OBS_VERSION_DOUBLES
    from lategame.model.actor_critic import ActorCritic
    from lategame.teambuilding.pool import DEFAULT_VGC_POOL
    from lategame.train.ppo import _save_checkpoint
    from lategame.train.selfplay import SelfPlayConfig, run_selfplay

    n_bins = 21
    init = tmp_path / "vgc_iter0.pt"
    # `_save_checkpoint` stamps input_dim/n_actions/obs_version from `codec_for(battle_format)`, so
    # the fixture cannot disagree with the codec the loop will build against -- which is the exact
    # mismatch `_check_warm_start` exists to catch, and would make this test pass vacuously.
    _save_checkpoint(
        ActorCritic(OBS_DIM_DOUBLES, hidden_dim=16, n_actions=214, n_bins=n_bins),
        str(init),
        VGC_FORMAT,
        -5.0,
        5.0,
        n_bins,
    )

    config = SelfPlayConfig(
        init=str(init),
        out_dir=str(tmp_path / "ckpts"),
        data_dir=str(tmp_path / "data"),
        battle_format=VGC_FORMAT,
        team_pool=str(DEFAULT_VGC_POOL),
        iters=1,
        games_per_opp=1,
        pop_size=1,
        buffer_iters=1,
        anchors=("heuristic",),
        eval_baselines=("random",),
        eval_n=1,
        epochs=1,
        device="cpu",
        # B6f's backstop. A VGC battle that loops does not end, and an unbounded one here would
        # hang the suite rather than fail it.
        max_battle_turns=60,
    )
    curve = await run_selfplay(config)

    assert (tmp_path / "ckpts" / "iter_01.pt").exists()
    assert [point["iter"] for point in curve] == [0, 1]

    import torch

    trained = torch.load(tmp_path / "ckpts" / "iter_01.pt", map_location="cpu", weights_only=False)
    assert trained["battle_format"] == VGC_FORMAT
    assert trained["obs_version"] == OBS_VERSION_DOUBLES
    assert trained["input_dim"] == OBS_DIM_DOUBLES
    assert trained["n_actions"] == 214, "two 107-way slot heads, not a singles 26"

    saved = json.loads((tmp_path / "ckpts" / "curve.json").read_text())
    assert [point["iter"] for point in saved] == [0, 1]
    assert "vs_random" in saved[1]

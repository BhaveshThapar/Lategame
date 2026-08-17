"""Unit tests for the M4 self-play building blocks (no server needed).

Covers the league sampler, the checkpoint-kwarg forwarding shared by ``build_player``
and the recording-player builder, the replay-buffer shard concatenation, and the
actor-critic -> actor-critic warm-start. The full loop is exercised by the
server-gated ``test_selfplay_smoke``.
"""

import random

import numpy as np
import pytest

from rotomai.data import collect
from rotomai.data.collect import PlayerSpec, TrajectoryDataset, concat_rl_shards, save_rl
from rotomai.data.reward import RewardWeights
from rotomai.eval import arena
from rotomai.features.action_space import GEN9_ACTION_SPACE_SIZE
from rotomai.features.encoder import OBS_DIM
from rotomai.train.selfplay import _sample_league


def test_sample_league_uniform_subset():
    league = ["a", "b", "c", "d"]
    picked = _sample_league(league, 2, random.Random(0))
    assert len(picked) == 2
    assert len(set(picked)) == 2  # distinct members
    assert set(picked) <= set(league)


def test_sample_league_handles_single_member():
    assert _sample_league(["only"], 3, random.Random(0)) == ["only"]


def test_build_player_forwards_checkpoint_kwargs(monkeypatch):
    captured: dict = {}

    class Dummy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(arena.AGENTS, "offrl", Dummy)
    arena.build_player("offrl", "gen9randombattle", checkpoint_path="x.pt", sample=True)
    assert captured["checkpoint_path"] == "x.pt"
    assert captured["sample"] is True


def test_build_player_omits_kwargs_for_baseline(monkeypatch):
    captured: dict = {}

    class Dummy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(arena.AGENTS, "random", Dummy)
    arena.build_player("random", "gen9randombattle", checkpoint_path="x.pt", sample=True)
    assert "checkpoint_path" not in captured
    assert "sample" not in captured


def test_recording_player_forwards_checkpoint_kwargs(monkeypatch):
    captured: dict = {}

    class Dummy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(collect.AGENTS, "offrl", Dummy)
    collect._build_recording_player(
        PlayerSpec("offrl", checkpoint_path="x.pt", sample=True), "gen9randombattle"
    )
    assert captured["checkpoint_path"] == "x.pt"
    assert captured["sample"] is True


def test_selfplay_config_has_the_teambuilt_and_cap_fields():
    """Both default to None, so every RB/OU arm ever run is bit-identical."""
    from rotomai.train.selfplay import SelfPlayConfig

    cfg = SelfPlayConfig()
    assert cfg.team_pool is None
    assert cfg.max_battle_turns is None


def test_run_selfplay_rejects_format_mismatch(tmp_path):
    """Collection reads the checkpoint's format, eval reads the config's; a mismatch means the
    two measure different games. `run_ppo` has refused this since Build 26; M4 did not."""
    import asyncio

    pytest.importorskip("torch")
    from rotomai.model.actor_critic import ActorCritic
    from rotomai.train.ppo import _save_checkpoint
    from rotomai.train.selfplay import SelfPlayConfig, run_selfplay

    path = tmp_path / "rb_init.pt"
    _save_checkpoint(
        ActorCritic(OBS_DIM, hidden_dim=16, n_bins=11), str(path), "gen9randombattle", -3.0, 3.0, 11
    )
    cfg = SelfPlayConfig(init=str(path), battle_format="gen9ou")
    with pytest.raises(ValueError, match="format mismatch"):
        asyncio.run(run_selfplay(cfg))


def test_run_selfplay_rejects_an_encoder_mismatched_warm_start(tmp_path):
    """A singles-dimension checkpoint tagged as VGC clears the format check and must die on the
    codec fingerprint -- otherwise it reaches the fine-tune step as a shape error, one full
    iteration of collection later."""
    import asyncio

    pytest.importorskip("torch")
    from rotomai.model.actor_critic import ActorCritic
    from rotomai.train.ppo import _save_checkpoint
    from rotomai.train.selfplay import SelfPlayConfig, run_selfplay

    path = tmp_path / "fake_vgc_init.pt"
    _save_checkpoint(
        ActorCritic(OBS_DIM, hidden_dim=16, n_bins=11), str(path), "gen9vgc2025regi", -3.0, 3.0, 11
    )
    cfg = SelfPlayConfig(init=str(path), battle_format="gen9vgc2025regi")
    with pytest.raises(ValueError, match="encoder mismatch"):
        asyncio.run(run_selfplay(cfg))


def test_run_selfplay_rejects_a_warm_start_with_no_value_support(tmp_path):
    """The guard's own blind spot: a BC checkpoint clears format AND encoder, then dies on
    `KeyError: 'v_min'` one statement later.

    `checkpoints/doubles_bc_vgc_v2.pt` is exactly this -- `n_bins` 51, no `v_min`/`v_max`, because
    BC fits no critic. Self-play pins the value support to its init, so there is nothing to pin to;
    the point of a pre-flight guard is that this says so by name instead of raising a bare KeyError
    from a dict access.
    """
    import asyncio

    import torch

    pytest.importorskip("torch")
    from rotomai.model.actor_critic import ActorCritic
    from rotomai.train.ppo import _save_checkpoint
    from rotomai.train.selfplay import SelfPlayConfig, run_selfplay

    path = tmp_path / "bc_shaped_init.pt"
    _save_checkpoint(
        ActorCritic(OBS_DIM, hidden_dim=16, n_bins=11), str(path), "gen9randombattle", -3.0, 3.0, 11
    )
    # Strip the support the way `train_bc` never writes it in the first place.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    del ckpt["v_min"], ckpt["v_max"]
    torch.save(ckpt, path)

    cfg = SelfPlayConfig(init=str(path), battle_format="gen9randombattle")
    with pytest.raises(ValueError, match="no value support"):
        asyncio.run(run_selfplay(cfg))


def test_selfplay_eval_points_build_the_agent_the_format_wants(monkeypatch):
    """`_eval_point` was the first thing a VGC self-play run hit, and it hardcoded `offrl`.

    `build_player` refuses a singles-only name on a doubles format, so `run_selfplay` raised at
    iteration 0's eval point -- before a battle was ever requested. Pins both directions: the
    doubles format gets a doubles-safe name, and the singles path is unchanged.
    """
    import asyncio

    from rotomai.train import selfplay as sp

    async def fake_eval(a, b, n):
        return 0.5

    def calls_for(fmt: str) -> list[tuple[str, dict]]:
        calls: list[tuple[str, dict]] = []

        def fake_build(name, battle_format, **kwargs):
            calls.append((name, kwargs))
            return object()

        monkeypatch.setattr(sp, "build_player", fake_build)
        monkeypatch.setattr(sp, "evaluate_built", fake_eval)
        asyncio.run(sp._eval_point(1, "ck.pt", sp.SelfPlayConfig(battle_format=fmt), team="POOL"))
        return calls

    for fmt, expected in (("gen9vgc2025regi", "doubles"), ("gen9randombattle", "offrl")):
        calls = calls_for(fmt)
        learner_names = {n for n, kw in calls if "checkpoint_path" in kw}
        assert learner_names == {expected}
        for name, _kw in calls:
            assert name in arena.AGENTS
            assert name not in arena._singles_only_agents() or fmt == "gen9randombattle"
        # Every player, baselines included: on a teambuilt format a baseline with no team
        # cannot start a battle, and only the learner builds were ever going to be looked at.
        assert all(kw.get("team") == "POOL" for _n, kw in calls)


# The literal "never hardcodes `offrl`" pin for this module now lives in
# `tests/test_arena.py::test_no_dispatch_module_hardcodes_the_singles_learner`, which checks the
# same thing for every module that builds a learned player from a caller-supplied format. Keeping a
# selfplay-only copy is how the enumeration drifted in the first place.


def test_collect_selfplay_gives_each_side_its_own_team_draw(monkeypatch):
    """A teambuilt M4 arm must reach BOTH sides, and the two must not draw in lockstep.

    ``collect_selfplay`` had no ``team_pool`` parameter at all, so a VGC self-play run had
    nowhere to put a pool even once ``SelfPlayConfig`` grew the field. Passing ONE pool object
    to both players would satisfy a naive "team is not None" check while making every battle a
    same-team mirror, so this pins that the two objects differ.
    """
    import asyncio

    from rotomai.teambuilding.pool import DEFAULT_VGC_POOL

    captured: list[dict] = []

    class Dummy:
        dropped = 0

    def fake_build(spec, battle_format, weights=None, max_concurrent=1, **kwargs):
        captured.append(kwargs)
        return Dummy()

    async def fake_cross_evaluate(players, n_challenges):
        return None

    def fake_append(player, weights, obs, action, mask, reward, done):
        obs.append(np.zeros((1, OBS_DIM), dtype=np.float32))
        action.append(np.zeros(1, dtype=np.int64))
        mask.append(np.ones((1, GEN9_ACTION_SPACE_SIZE), dtype=bool))
        reward.append(np.zeros(1, dtype=np.float32))
        done.append(np.array([True]))
        return 0

    monkeypatch.setattr(collect, "_build_recording_player", fake_build)
    monkeypatch.setattr(collect, "cross_evaluate", fake_cross_evaluate)
    monkeypatch.setattr(collect, "_append_episodes", fake_append)

    asyncio.run(
        collect.collect_selfplay(
            PlayerSpec("offrl"),
            [PlayerSpec("simpleheuristics")],
            1,
            "gen9ou",
            team_pool=str(DEFAULT_VGC_POOL),
        )
    )

    teams = [c["team"] for c in captured]
    assert len(teams) == 2
    assert all(t is not None for t in teams)
    assert teams[0] is not teams[1]


def test_collect_selfplay_leaves_random_battles_teamless(monkeypatch):
    """Random Battles must NOT get a team -- the server supplies them there."""
    import asyncio

    captured: list[dict] = []

    class Dummy:
        dropped = 0

    def fake_build(spec, battle_format, weights=None, max_concurrent=1, **kwargs):
        captured.append(kwargs)
        return Dummy()

    async def fake_cross_evaluate(players, n_challenges):
        return None

    def fake_append(player, weights, obs, action, mask, reward, done):
        obs.append(np.zeros((1, OBS_DIM), dtype=np.float32))
        action.append(np.zeros(1, dtype=np.int64))
        mask.append(np.ones((1, GEN9_ACTION_SPACE_SIZE), dtype=bool))
        reward.append(np.zeros(1, dtype=np.float32))
        done.append(np.array([True]))
        return 0

    monkeypatch.setattr(collect, "_build_recording_player", fake_build)
    monkeypatch.setattr(collect, "cross_evaluate", fake_cross_evaluate)
    monkeypatch.setattr(collect, "_append_episodes", fake_append)

    asyncio.run(
        collect.collect_selfplay(PlayerSpec("offrl"), [PlayerSpec("random")], 1, "gen9randombattle")
    )
    assert [c["team"] for c in captured] == [None, None]


def _tiny_traj(n: int, val: float = 1.0) -> TrajectoryDataset:
    return TrajectoryDataset(
        obs=np.zeros((n, OBS_DIM), dtype=np.float32),
        action=np.zeros(n, dtype=np.int64),
        mask=np.ones((n, GEN9_ACTION_SPACE_SIZE), dtype=bool),
        reward=np.full(n, val, dtype=np.float32),
        done=np.array([False] * (n - 1) + [True]),
        battle_format="gen9randombattle",
        gamma=1.0,
        weights=RewardWeights(),
    )


def test_concat_rl_shards_preserves_episodes(tmp_path):
    s1, s2 = tmp_path / "a.npz", tmp_path / "b.npz"
    save_rl(_tiny_traj(2), s1)
    save_rl(_tiny_traj(3), s2)
    ds = concat_rl_shards([s1, s2], tmp_path / "buf.npz")
    assert len(ds) == 5
    assert int(ds.done.sum()) == 2  # one terminal per shard, preserved across the seam


def test_concat_rl_shards_returns_reset_across_seam(tmp_path):
    pytest.importorskip("torch")
    from rotomai.data.rl_dataset import RLDataset

    s1, s2 = tmp_path / "a.npz", tmp_path / "b.npz"
    save_rl(_tiny_traj(2), s1)
    save_rl(_tiny_traj(3), s2)
    out = tmp_path / "buf.npz"
    concat_rl_shards([s1, s2], out)
    # gamma=1: shard1 returns [2,1], shard2 [3,2,1] -- the scan must reset at the seam.
    np.testing.assert_allclose(RLDataset(out).ret.numpy(), [2, 1, 3, 2, 1])


def test_load_actor_critic_weights_roundtrip():
    torch = pytest.importorskip("torch")
    from rotomai.model.actor_critic import ActorCritic, load_actor_critic_weights

    src = ActorCritic(OBS_DIM, hidden_dim=16, n_bins=11)
    dst = ActorCritic(OBS_DIM, hidden_dim=16, n_bins=11)
    load_actor_critic_weights(dst, src.state_dict())
    for key, tensor in src.state_dict().items():
        assert torch.equal(tensor, dst.state_dict()[key])


def test_load_actor_critic_weights_shape_mismatch_raises():
    pytest.importorskip("torch")
    from rotomai.model.actor_critic import ActorCritic, load_actor_critic_weights

    src = ActorCritic(OBS_DIM, hidden_dim=16, n_bins=11)
    dst = ActorCritic(OBS_DIM, hidden_dim=16, n_bins=21)  # value head shape differs
    with pytest.raises(RuntimeError):
        load_actor_critic_weights(dst, src.state_dict())


def test_curriculum_gate_scopes_its_paths_to_the_arm():
    """`arm`, `checkpoints/<arm>_s<seed>/` and `data/` were the literal `curriculum_et_prior`, so a
    VGC arm would have overwritten the Random Battles one in place and produced a record whose
    `arm` field named the wrong experiment."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "curriculum_gate.py"
    spec = importlib.util.spec_from_file_location("curriculum_gate", path)
    assert spec is not None and spec.loader is not None
    cg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cg)

    assert cg._out_dir("m4_vgc", 1) == "checkpoints/m4_vgc_s1"
    # The Random Battles arm keeps its historical paths byte for byte.
    assert cg._out_dir("curriculum_et_prior", 0) == "checkpoints/curriculum_et_prior_s0"


def test_collect_selfplay_refuses_a_teambuilt_format_with_no_team_pool():
    """Without a pool the failure is NOT an error, which is why it needs to be made one.

    Showdown answers a teamless challenge on a teambuilt format with a popup -- "Your team was
    rejected ... This format requires you to use your own team" -- that poke-env logs as a WARNING.
    Collection then gathers nothing while every surface reads as running. The first VGC M4 job died
    exactly this way, after claiming its node, because `curriculum_gate`'s Gate A collects through
    its own `collect_selfplay` call and the pool had only been threaded into `SelfPlayConfig`.
    """
    import asyncio

    for fmt in ("gen9vgc2025regi", "gen9ou"):
        with pytest.raises(ValueError, match="needs a `team_pool`"):
            asyncio.run(
                collect.collect_selfplay(PlayerSpec("doubles"), [PlayerSpec("random")], 1, fmt)
            )


def test_curriculum_gate_preflight_takes_the_pool_and_the_turn_cap():
    """Gate A is a separate collection from the self-play loop's, so `SelfPlayConfig` carrying the
    pool does nothing for it. Pinned by signature so the two cannot drift apart again."""
    import importlib.util
    import inspect
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "curriculum_gate.py"
    spec = importlib.util.spec_from_file_location("curriculum_gate", path)
    assert spec is not None and spec.loader is not None
    cg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cg)

    params = inspect.signature(cg.run_preflight).parameters
    assert "team_pool" in params and "max_battle_turns" in params
    assert "team_pool=team_pool" in inspect.getsource(cg.run_preflight)
